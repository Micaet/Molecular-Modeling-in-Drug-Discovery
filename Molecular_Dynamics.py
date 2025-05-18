import openmm as mm
from openmm import app
from openmm import unit
from sys import stdout
from pdbfixer import PDBFixer
import xml.etree.ElementTree as ET
import os
import mdtraj as md
from mdtraj.reporters import XTCReporter
import matplotlib.pyplot as plt
import numpy as np
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.typing.engines.smirnoff import ForceField as OpenFFForceField

# Lista struktur do symulacji
structure_ids = ['8','15','66','124','313']

# Funkcja do tworzenia ścieżek dla danej struktury
def get_paths(structure_id):
    base_input = f'dynamics/{structure_id}/input'
    base_output = f'dynamics/{structure_id}/output'
    return {
        'fixer_pdb': f'{base_input}/5cno_prepared.pdb',
        'fixed_protein': f'{base_input}/protein_fixed.pdb',
        'protein_amber_H': f'{base_input}/protein_amber_H.pdb',
        'ligand': f'{base_input}/ligand{structure_id}_h.sdf',
        'output_dir': f'{base_output}/simulation_main'
    }

# --- Function to run simulation ---
def run_simulation_logic(sim_output_dir, sim_integrator, sim_topology, sim_positions, sim_system, include_heating, total_steps_production):
    print(f"--- Starting Simulation Run: Output to {sim_output_dir} ---")
    
    local_simulation = app.Simulation(sim_topology, sim_system, sim_integrator)
    local_simulation.context.setPositions(sim_positions)

    print(f"Minimizing energy for run: {sim_output_dir}...")
    local_simulation.minimizeEnergy()
    # Pobierz pozycje po lokalnej minimalizacji do zapisu w PDB
    positions_after_local_minimization = local_simulation.context.getState(getPositions=True).getPositions()
    # sim_topology to topologia przekazana do funkcji, która jest poprawna dla tego systemu

    # --- Konfiguracja Reporterów ---
    pdb_reporter_path = os.path.join(sim_output_dir, 'trajectory.pdb')
    xtc_reporter_path = os.path.join(sim_output_dir, 'trajectory.xtc')
    data_reporter_path = os.path.join(sim_output_dir, 'simulation_data.txt')
    # final_state_path = os.path.join(sim_output_dir, 'final_state.pdb') # Zakomentowane

    # --- Bezpośredni zapis klatki początkowej PDB ---
    print(f"Writing initial PDB frame (after local minimization) to: {pdb_reporter_path}")
    with open(pdb_reporter_path, 'w') as f_pdb:
        app.PDBFile.writeFile(sim_topology, positions_after_local_minimization, f_pdb)
    # --- Koniec bezpośredniego zapisu PDB ---

    # Usunięto PDBReporter dla trajectory.pdb, ponieważ zapisujemy go bezpośrednio powyżej.
    # local_simulation.reporters.append(app.PDBReporter(pdb_reporter_path, pdb_report_interval))
    
    # Reporter XTC pozostaje bez zmian
    local_simulation.reporters.append(XTCReporter(xtc_reporter_path, 1000))
    
    report_file = open(data_reporter_path, 'w')
    local_simulation.reporters.append(app.StateDataReporter(report_file, 1000, step=True, potentialEnergy=True, temperature=True))
    
    # Obliczanie effective_total_steps dla reportera stdout.
    # Wartość actual_heating_duration_steps jest nadal potrzebna.
    actual_heating_duration_steps = 0
    if include_heating:
        actual_heating_duration_steps = 25000  # Całkowita liczba kroków dla fazy ogrzewania

    effective_total_steps = total_steps_production
    if include_heating:
        effective_total_steps += actual_heating_duration_steps # Dodajemy kroki ogrzewania

    local_simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, progress=True, remainingTime=True, speed=True, totalSteps=effective_total_steps))

    if include_heating:
        print('Starting heating phase...')
        initial_temp = 50*unit.kelvin
        final_temp = 300*unit.kelvin
        # heating_steps_val is the total duration for heating in terms of simulation steps
        heating_steps_val = 25000  # This matches actual_heating_duration_steps

        # User wants approximately 1K increment from 50K to 300K.
        # Total temperature range is 250K (300K - 50K). So, 250 stages.
        n_stages = 250 
        
        if n_stages <= 0: 
            # This case should ideally not be reached with fixed 50K->300K and n_stages=250
            steps_per_stage = heating_steps_val 
            if n_stages < 0: n_stages = 0 # Prevent negative range in loop
        else:
            steps_per_stage = heating_steps_val // n_stages

        print(f"Heating from {initial_temp} to {final_temp} over {heating_steps_val} steps, in {n_stages} stages of {steps_per_stage} steps each.")

        for i in range(n_stages):
            # Calculate target temperature for the current stage
            current_stage_fraction = (i + 1) / n_stages
            temp_k_val = initial_temp.value_in_unit(unit.kelvin) + \
                         (final_temp.value_in_unit(unit.kelvin) - initial_temp.value_in_unit(unit.kelvin)) * current_stage_fraction
            temp = temp_k_val * unit.kelvin
            
            sim_integrator.setTemperature(temp)
            local_simulation.step(steps_per_stage)
            # Report the target temperature for the stage
            # print(f"Heating stage {i+1}/{n_stages} completed. Integrator temperature set to {temp.value_in_unit(unit.kelvin):.2f} K for {sim_output_dir}")
        
        sim_integrator.setTemperature(final_temp) # Ensure final target temperature is set after all stages
        print('Heating phase completed.')

    print(f'Running production simulation for {sim_output_dir}...')
    local_simulation.step(total_steps_production)

    # --- Zapisanie Stanu Końcowego --- (Zakomentowane)
    # final_positions = local_simulation.context.getState(getPositions=True).getPositions()
    # app.PDBFile.writeFile(local_simulation.topology, final_positions, open(final_state_path, 'w'))
    report_file.close()
    print(f"--- Simulation Run Completed: {sim_output_dir} ---")

# --- Function for Analysis and Plotting ---
def analyze_trajectory_and_plot_histograms(
    xtc_file_path,
    pdb_topology_file_path,
    output_prefix,
    protein_omm_topology_for_vac_calc, # Added: Protein OpenMM Topology
    protein_omm_positions_for_vac_calc, # Added: Protein OpenMM Positions
    ligand_openff_molecule_for_vac_calc, # Added: Ligand OpenFF Molecule
    ligand_omm_topology_for_vac_calc     # Added: Ligand OpenMM Topology
):
    print(f"\n--- Analyzing Trajectory: {xtc_file_path} ---")
    try:
        traj = md.load(xtc_file_path, top=pdb_topology_file_path) 
        print(f"Loaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms.")

        if traj.n_frames == 0:
            print(f"Warning: Trajectory file {xtc_file_path} contains 0 frames. Skipping all trajectory analysis for this run.")
            # Ensure any matplotlib figures are closed if we exit early from analysis
            # This avoids potential issues with leftover plot states.
            plt.close('all') 
            return # Exit analysis for this trajectory

        protein_ca_indices = traj.topology.select('protein and name CA')
        if protein_ca_indices.size == 0:
            print("Warning: Could not find protein C-alpha atoms. Skipping RMSD and RMSF analysis.")
        else:
            print("Calculating RMSD...")
            rmsd = md.rmsd(traj, traj, frame=0, atom_indices=protein_ca_indices)
            plt.figure()
            plt.hist(rmsd * 10, bins=50) 
            plt.xlabel('RMSD (Å)')
            plt.ylabel('Frequency')
            plt.title(f'Protein C-alpha RMSD (vs. first frame)\n{os.path.basename(output_prefix)}')
            plt.savefig(f"{output_prefix}_protein_ca_rmsd_histogram.png")
            plt.clf()
            print(f"Saved RMSD histogram to {output_prefix}_protein_ca_rmsd_histogram.png")

            print("Calculating RMSF...")
            traj.superpose(traj, frame=0, atom_indices=protein_ca_indices)
            rmsf_per_atom = md.rmsf(traj, traj, frame=0, atom_indices=protein_ca_indices) 
            
            plt.figure()
            plt.plot(rmsf_per_atom * 10) # Convert nm to Angstrom
            plt.xlabel('C-alpha Atom Index (within selection)')
            plt.ylabel('RMSF (Å)')
            plt.title(f'Protein C-alpha RMSF\n{os.path.basename(output_prefix)}')
            plt.savefig(f"{output_prefix}_protein_ca_rmsf_plot.png")
            plt.clf()
            print(f"Saved RMSF plot to {output_prefix}_protein_ca_rmsf_plot.png")

        protein_indices = traj.topology.select('protein')
        if protein_indices.size == 0:
            print("Warning: Could not find protein atoms. Skipping SASA analysis.")
        else:
            print("Calculating SASA...")
            sasa_traj_slice = traj[100::25] if traj.n_frames > 100 else traj
            
            sasa_per_atom = md.shrake_rupley(sasa_traj_slice, mode='atom')
            protein_sasa_per_atom_per_frame = sasa_per_atom[:, protein_indices]
            total_sasa_per_frame = np.sum(protein_sasa_per_atom_per_frame, axis=1)
            
            plt.figure()
            plt.hist(total_sasa_per_frame, bins=50)
            plt.xlabel('Total Protein SASA (nm^2)')
            plt.ylabel('Frequency')
            plt.title(f'Total Protein SASA\n{os.path.basename(output_prefix)}')
            plt.savefig(f"{output_prefix}_protein_sasa_histogram.png")
            plt.clf()
            print(f"Saved SASA histogram to {output_prefix}_protein_sasa_histogram.png")
        

        # --- Calculate MM Interaction Energy (Simplified Delta G component) ---
        print("\n--- Calculating MM Interaction Energy ---")
        try:
            # 1. Prepare vacuum systems
            # Protein System (AMBER FF)
            protein_ff = app.ForceField('amber14-all.xml')
            protein_vac_system = protein_ff.createSystem(protein_omm_topology_for_vac_calc,
                                                         nonbondedMethod=app.NoCutoff,
                                                         constraints=None,
                                                         rigidWater=False)
            
            # Ligand System (OpenFF FF)
            ligand_ff_str = 'openff_unconstrained-2.1.0.offxml' # As used in main simulation
            ligand_openff_ff_vac = OpenFFForceField(ligand_ff_str, allow_cosmetic_attributes=True)


            ligand_vac_system = ligand_openff_ff_vac.create_openmm_system(
                ligand_openff_molecule_for_vac_calc.to_topology(), # OpenFF Topology
                charge_from_molecules=[ligand_openff_molecule_for_vac_calc],
            )
            for force in ligand_vac_system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                    force.setUseDispersionCorrection(False)
            
            # Complex System (Protein:AMBER, Ligand:OpenFF)
            # Create a new Modeller for the complex in vacuum
            complex_vac_modeller = app.Modeller(protein_omm_topology_for_vac_calc, protein_omm_positions_for_vac_calc)
            complex_vac_modeller.add(ligand_omm_topology_for_vac_calc, ligand_openff_molecule_for_vac_calc.conformers[0].to_openmm())
            complex_vac_topology = complex_vac_modeller.getTopology()

            complex_system_generator_vac = SystemGenerator(
                forcefields=['amber14-all.xml'], 
                small_molecule_forcefield=ligand_ff_str,
                molecules=[ligand_openff_molecule_for_vac_calc],
                cache='openff_cache.sqlite' # Can reuse cache
            )
            complex_vac_system = complex_system_generator_vac.create_system(
                complex_vac_topology
            )
            # Ręczne ustawienie NoCutoff dla NonbondedForce w complex_vac_system
            for force in complex_vac_system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                    # Usuwamy także PME, jeśli byłby domyślnie ustawiony dla NonbondedForce przez SystemGenerator
                    # (choć dla NoCutoff to nie powinno mieć znaczenia, ale dla pewności)
                    force.setUseDispersionCorrection(False) # Często wyłączane przy NoCutoff dla czystej energii
                    if force.getNonbondedMethod() == mm.NonbondedForce.PME:
                        # To nie powinno się zdarzyć, jeśli ustawiliśmy NoCutoff,
                        # ale na wszelki wypadek, jeśli jakaś logika by to przywróciła
                        print("Warning: PME was set on complex_vac_system's NonbondedForce despite NoCutoff request. Attempting to ensure NoCutoff.")
            
            # Usuwanie ograniczeń (constraints) z complex_vac_system, jeśli są obecne
            # SystemGenerator może je dodać na podstawie pól siłowych
            num_constraints_complex_vac = complex_vac_system.getNumConstraints()
            if num_constraints_complex_vac > 0:
                print(f"Note: Complex vacuum system initially has {num_constraints_complex_vac} constraints. Removing them for pure potential energy calculation.")
                # Aby usunąć, musimy stworzyć nowy system bez nich, bo nie ma metody removeAllConstraints
                # Ale dla celów obliczenia energii, constraints nie wpływają na *potencjalną* energię, tylko na dynamikę.
                # Na razie zostawiamy, ale jeśli wyniki będą dziwne, można tu wrócić.
                # Alternatywnie, `SystemGenerator` ma argument `constraints=None`, ale może nie być respektowany dla wszystkich pól siłowych.
                pass # Na razie nie usuwamy, bo nie wpływają na E_pot.

            # 2. Atom selection from trajectory (which is solvated)
            # First, get the ligand residue name. This is crucial for correct protein selection.
            ligand_resname_in_traj = "LIG" # Default
            if ligand_omm_topology_for_vac_calc.getNumResidues() > 0:
                first_lig_res = list(ligand_omm_topology_for_vac_calc.residues())[0]
                if first_lig_res.name:
                    ligand_resname_in_traj = first_lig_res.name
            
            # Select protein atoms, EXCLUDING the ligand
            protein_selection_string = f'(protein and not water) and not (resname "{ligand_resname_in_traj}")'
            protein_indices_traj = traj.topology.select(protein_selection_string)
            # print(f"Protein selection string: '{protein_selection_string}', selected atoms: {protein_indices_traj.size}")
            
            # Select ligand atoms
            ligand_selection_string = f'(resname "{ligand_resname_in_traj}") and (not water)'
            ligand_indices_traj = traj.topology.select(ligand_selection_string)
            # print(f"Ligand selection string: '{ligand_selection_string}', selected atoms: {ligand_indices_traj.size}")

            if ligand_indices_traj.size == 0 and ligand_openff_molecule_for_vac_calc.name:
                 mol_name = ligand_openff_molecule_for_vac_calc.name.strip()
                 if mol_name:
                    ligand_selection_string_fallback = f'(resname "{mol_name}") and (not water)'
                    # print(f"Warning: Ligand selection by primary resname '{ligand_resname_in_traj}' failed. Trying fallback: '{ligand_selection_string_fallback}'")
                    ligand_indices_traj = traj.topology.select(ligand_selection_string_fallback)
                    if ligand_indices_traj.size > 0:
                        ligand_resname_in_traj = mol_name # Update if fallback successful for protein exclusion
                        protein_selection_string = f'(protein and not water) and not (resname "{ligand_resname_in_traj}")'
                        protein_indices_traj = traj.topology.select(protein_selection_string)
                        # print(f"Protein selection updated after ligand fallback: '{protein_selection_string}', selected atoms: {protein_indices_traj.size}")
                        # print(f"Ligand selection by fallback '{ligand_selection_string_fallback}' successful, selected atoms: {ligand_indices_traj.size}")

            if ligand_indices_traj.size == 0: # Final fallback if still no ligand atoms
                print(f"Critical Warning: Could not select ligand by resname '{ligand_resname_in_traj}' or molecule name. Interaction energy calculation will likely fail or be incorrect.")
                # Allow to proceed to see further errors if any, but this is a bad state.

            # Atom count validation
            err_msg_atom_count = ""
            if not protein_omm_topology_for_vac_calc.getNumAtoms() == protein_indices_traj.size:
                err_msg_atom_count += f"Protein atom count mismatch: VAC_TOPO={protein_omm_topology_for_vac_calc.getNumAtoms()}, TRAJ_SELECT={protein_indices_traj.size}.\n"
            if not ligand_omm_topology_for_vac_calc.getNumAtoms() == ligand_indices_traj.size:
                err_msg_atom_count += f"Ligand atom count mismatch: VAC_TOPO={ligand_omm_topology_for_vac_calc.getNumAtoms()}, TRAJ_SELECT={ligand_indices_traj.size} (Resname used: '{ligand_resname_in_traj}').\n"
            if not complex_vac_topology.getNumAtoms() == (protein_indices_traj.size + ligand_indices_traj.size):
                 err_msg_atom_count += f"Complex atom count mismatch: VAC_TOPO={complex_vac_topology.getNumAtoms()}, TRAJ_SUM_SELECT={protein_indices_traj.size + ligand_indices_traj.size}.\n"

            if err_msg_atom_count:
                print("ERROR: Atom count mismatch between trajectory selection and vacuum topologies.")
                print(err_msg_atom_count)
                raise ValueError("Atom count mismatch for interaction energy calculation. Check ligand selection logic.")

            # 3. Integrator and Contexts
            # Każdy kontekst potrzebuje własnej instancji integratora.
            # dummy_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds) # Usunięte - tworzymy osobne dla każdego kontekstu
            
            # print("Creating vacuum contexts for energy calculation...")
            protein_vac_context = mm.Context(protein_vac_system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
            # print("Protein vacuum context created.")
            ligand_vac_context = mm.Context(ligand_vac_system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
            # print("Ligand vacuum context created.")
            complex_vac_context = mm.Context(complex_vac_system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
            # print("Complex vacuum context created.")

            interaction_energies_kj_mol = []
            
            num_frames_for_ie = min(50, traj.n_frames // 20 if traj.n_frames >= 100 else traj.n_frames) # e.g. 50 frames, or 5% (min 1 frame)
            if num_frames_for_ie == 0 and traj.n_frames > 0: num_frames_for_ie = 1 
            
            if num_frames_for_ie > 0:
                frame_indices_for_ie = np.linspace(0, traj.n_frames - 1, num_frames_for_ie, dtype=int)
                selected_traj_for_ie = traj[frame_indices_for_ie]
                print(f"Calculating interaction energies for {selected_traj_for_ie.n_frames} frames...")

                for frame_idx, frame in enumerate(selected_traj_for_ie):
                    # print(f"  Processing frame {frame_idx+1}/{selected_traj_for_ie.n_frames} for IE...")
                    frame_xyz_nm = frame.xyz[0]

                    protein_pos_nm = frame_xyz_nm[protein_indices_traj]
                    ligand_pos_nm = frame_xyz_nm[ligand_indices_traj]
                    
                    complex_pos_nm = np.concatenate((protein_pos_nm, ligand_pos_nm), axis=0)

                    protein_vac_context.setPositions(protein_pos_nm * unit.nanometer)
                    E_prot = protein_vac_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                    ligand_vac_context.setPositions(ligand_pos_nm * unit.nanometer)
                    E_lig = ligand_vac_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    
                    complex_vac_context.setPositions(complex_pos_nm * unit.nanometer)
                    E_complex = complex_vac_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                    interaction_energy = E_complex - (E_prot + E_lig)
                    interaction_energies_kj_mol.append(interaction_energy)
            else:
                print("No frames selected for interaction energy calculation (trajectory too short or num_frames_for_ie=0).")

            if interaction_energies_kj_mol:
                avg_ie = np.mean(interaction_energies_kj_mol)
                std_ie = np.std(interaction_energies_kj_mol)
                print(f"Average MM Interaction Energy: {avg_ie:.2f} +/- {std_ie:.2f} kJ/mol")

                plt.figure()
                plt.hist(interaction_energies_kj_mol, bins=max(15, len(interaction_energies_kj_mol)//2), density=True) # Adjusted bins
                plt.xlabel('MM Interaction Energy (kJ/mol)')
                plt.ylabel('Frequency')
                title_text = (
                    f'Ligand-Protein MM Interaction Energy\\n'
                    f'{os.path.basename(output_prefix)}\\n'
                    f'Avg: {avg_ie:.2f} ± {std_ie:.2f} kJ/mol'
                )
                plt.title(title_text)
                plt.savefig(f"{output_prefix}_interaction_energy_histogram.png")
                plt.clf()
                print(f"Saved interaction energy histogram to {output_prefix}_interaction_energy_histogram.png")
            else:
                print("No interaction energies were calculated.")
            
        except Exception as e_ie:
            print(f"Error during MM Interaction Energy calculation: {e_ie}")
            import traceback
            traceback.print_exc()
        
        plt.close('all') 

    except Exception as e:
        print(f"Error during analysis of {xtc_file_path}: {e}")
    print(f"--- Analysis Completed for: {xtc_file_path} ---")

# Główna pętla po strukturach
for structure_id in structure_ids:
    print(f"\n{'='*50}")
    print(f"Processing structure {structure_id}")
    print(f"{'='*50}\n")
    
    # Pobierz ścieżki dla aktualnej struktury
    paths = get_paths(structure_id)
    
    # Utwórz katalog wyjściowy
    os.makedirs(paths['output_dir'], exist_ok=True)

    # --- 1. Inicjalizacja ---
    fixer = PDBFixer(paths['fixer_pdb'])
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False) 
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(paths['fixed_protein'], 'w'))
    print(f"Protein structure initially processed by PDBFixer (without addMissingHydrogens) saved to: {paths['fixed_protein']}")

    # --- 2. Przygotowanie wodorów w białku zgodnie z AMBER ---
    print("\nPreparing protein hydrogens using AMBER force field definitions...")
    protein_structure_for_H_fix = app.PDBFile(paths['fixed_protein'])

    protein_modeller = app.Modeller(protein_structure_for_H_fix.topology, protein_structure_for_H_fix.positions)
    protein_amber_ff = app.ForceField('amber14-all.xml')
    atoms_to_remove_from_protein = [atom for atom in protein_modeller.topology.atoms() if atom.element.symbol == 'H']
    if atoms_to_remove_from_protein:
        protein_modeller.delete(atoms_to_remove_from_protein)
        print(f"  Removed {len(atoms_to_remove_from_protein)} existing hydrogen atoms from protein.")
    else:
        print("  No hydrogen atoms found in protein to remove (expected if PDBFixer didn't add them or input had none).")

    print("  Adding AMBER-compliant hydrogens to protein...")
    protein_modeller.addHydrogens(protein_amber_ff, pH=7.0)
    app.PDBFile.writeFile(protein_modeller.topology, protein_modeller.positions, open(paths['protein_amber_H'], 'w'))
    print(f"  Protein with AMBER-compliant hydrogens saved to: {paths['protein_amber_H']}")

    protein_topology_final = protein_modeller.getTopology()
    protein_positions_final = protein_modeller.getPositions()

    # --- 3. Wczytanie struktur ---
    print(f"\nLoading ligand from: {paths['ligand']} for OpenFF parameterization...")
    try:
        ligand_off_molecule = Molecule.from_file(paths['ligand'], allow_undefined_stereo=True)
        print("Ligand loaded into OpenFF Molecule object.")
    except Exception as e:
        print(f"Error loading ligand {paths['ligand']} with OpenFF: {e}")
        print("Ensure the file path is correct and the format (PDBQT, SDF, MOL2) is readable by OpenFF.")
        print("For PDBQT, ensure it contains connectivity information or consider converting to SDF or MOL2 first.")
        raise

    print("Assigning partial charges to ligand using AM1BCCELF10...")
    try:
        ligand_off_molecule.assign_partial_charges(partial_charge_method='gasteiger')
        print("Partial charges assigned to ligand.")
    except Exception as e:
        print(f"Error assigning charges to ligand: {e}")
        print("Ensure you have 'openff-nagl' installed (e.g., via 'pip install openff-toolkit[charges]').")
        raise

    # --- 4. Połączenie systemu ---
    modeller = app.Modeller(protein_topology_final, protein_positions_final)
    ligand_omm_topology = ligand_off_molecule.to_topology().to_openmm()
    ligand_omm_positions = ligand_off_molecule.conformers[0].to_openmm()

    modeller.add(ligand_omm_topology, ligand_omm_positions)
    print("Ligand added to OpenMM Modeller.")

    # --- 5. Solwatacja i konfiguracja symulacji ---
    solvation_modeller = app.Modeller(protein_topology_final, protein_positions_final)
    solvent_ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    solvation_modeller.addSolvent(
        solvent_ff,
        padding=1.0*unit.nanometers,
        ionicStrength=0.15*unit.molar
    )

    solvated_topology = solvation_modeller.getTopology()
    solvated_positions = solvation_modeller.getPositions()
    modeller = app.Modeller(solvated_topology, solvated_positions)
    modeller.add(ligand_omm_topology, ligand_omm_positions)

    system_generator = SystemGenerator(
        forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
        small_molecule_forcefield='openff_unconstrained-2.1.0.offxml',
        molecules=[ligand_off_molecule],
        cache='openff_cache.sqlite'
    )
    final_system = system_generator.create_system(modeller.topology)

    # --- 6. Integrator ---
    temperature = 300*unit.kelvin
    friction = 1.0/unit.picosecond
    timestep = 2.0*unit.femtoseconds
    integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)

    # --- 7. Ustawienie symulacji ---
    simulation = app.Simulation(modeller.topology, final_system, integrator)
    simulation.context.setPositions(modeller.positions)

    # --- 8. Minimalizacja energii ---
    print('Minimizing energy...')
    simulation.minimizeEnergy()
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    minimized_topology = modeller.topology 

    # --- 9. Uruchomienie symulacji ---
    print("\nRunning Simulation WITH Heating Phase...")
    integrator_with_heating = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
    run_simulation_logic(
        sim_output_dir=paths['output_dir'],
        sim_integrator=integrator_with_heating,
        sim_topology=minimized_topology,
        sim_positions=minimized_positions,
        sim_system=final_system,
        include_heating=True,
        total_steps_production=5000000  # Zwiększone 100x z 50000
    )

    # --- 10. Analiza trajektorii ---
    analyze_trajectory_and_plot_histograms(
        xtc_file_path=os.path.join(paths['output_dir'], 'trajectory.xtc'),
        pdb_topology_file_path=os.path.join(paths['output_dir'], 'trajectory.pdb'),
        output_prefix=os.path.join(paths['output_dir'], 'analysis'),
        protein_omm_topology_for_vac_calc=protein_topology_final,
        protein_omm_positions_for_vac_calc=protein_positions_final, # Pass protein positions
        ligand_openff_molecule_for_vac_calc=ligand_off_molecule,
        ligand_omm_topology_for_vac_calc=ligand_omm_topology
    )

    print(f'\nCompleted processing structure {structure_id}')
    print(f"{'='*50}\n")

print('\nAll simulations completed!')