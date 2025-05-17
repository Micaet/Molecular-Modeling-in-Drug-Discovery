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
structure_ids = ['8', '15', '66', '124', '313']

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
    current_positions = local_simulation.context.getState(getPositions=True).getPositions()
    local_simulation.context.setPositions(current_positions)

    # --- Konfiguracja Reporterów ---
    pdb_reporter_path = os.path.join(sim_output_dir, 'trajectory.pdb')
    xtc_reporter_path = os.path.join(sim_output_dir, 'trajectory.xtc')
    data_reporter_path = os.path.join(sim_output_dir, 'simulation_data.txt')
    final_state_path = os.path.join(sim_output_dir, 'final_state.pdb')

    local_simulation.reporters.append(app.PDBReporter(pdb_reporter_path, 1000))
    local_simulation.reporters.append(XTCReporter(xtc_reporter_path, 1000))
    
    report_file = open(data_reporter_path, 'w')
    local_simulation.reporters.append(app.StateDataReporter(report_file, 1000, step=True, potentialEnergy=True, temperature=True))
    # Obliczanie total_steps dla reportera stdout
    effective_total_steps = total_steps_production
    if include_heating:
        # Zakładając, że heating_steps jest zdefiniowane globalnie lub przekazane inaczej
        # Dla uproszczenia, przyjmijmy, że heating_steps jest równe 25000 * 100 (zgodnie z wcześniejszymi zmianami)
        # Lepszym rozwiązaniem byłoby przekazanie heating_steps do funkcji
        # lub obliczenie go na podstawie n_stages i steps_per_stage, jeśli są stałe.
        # Tutaj użyjemy wartości bezpośrednio, ale pamiętaj, że to może wymagać dostosowania
        # jeśli logika heating_steps się zmieni.
        # Wcześniej: heating_steps = 2500000
        # Wcześniej: heating_steps = 25000
        # Użyjemy heating_steps = 25000, bo tak jest teraz w kodzie. Jesli ma być 100x, to trzeba to uwzględnić
        # Zgodnie z ostatnimi zmianami heating_steps było 2500000, ale potem zostało przywrócone do 25000 w diffie.
        # Będę zakładał, że heating_steps wewnątrz tej funkcji powinno odzwierciedlać to, co faktycznie będzie użyte.
        # Na podstawie kodu, który wykonuje pętlę ogrzewania:
        # heating_steps_in_loop = n_stages * steps_per_stage
        # W kodzie jest: heating_steps = 25000 (z diffa), n_stages = 50, steps_per_stage = heating_steps // n_stages = 500
        # więc faktyczne kroki ogrzewania to 50 * 500 = 25000
        effective_total_steps += 25000 # Dodajemy kroki ogrzewania

    local_simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, progress=True, remainingTime=True, speed=True, totalSteps=effective_total_steps))

    if include_heating:
        print('Starting heating phase...')
        initial_temp = 50*unit.kelvin
        final_temp = 300*unit.kelvin
        # Te wartości powinny być spójne z tym, co jest używane do obliczenia effective_total_steps
        heating_steps_val = 25000  # Zgodnie z ostatnim diffem, użytkownik przywrócił to do 25000
        n_stages = 50 
        steps_per_stage = heating_steps_val // n_stages

        for i in range(n_stages):
            temp = initial_temp + (final_temp - initial_temp) * (i+1)/n_stages
            sim_integrator.setTemperature(temp)
            local_simulation.step(steps_per_stage)
            print(f"Heating stage {i+1}/{n_stages} completed. Current temperature: {temp.value_in_unit(unit.kelvin):.2f} K for {sim_output_dir}")
        
        sim_integrator.setTemperature(final_temp) 
        print('Heating phase completed.')

    print(f'Running production simulation for {sim_output_dir}...')
    local_simulation.step(total_steps_production)

    # --- Zapisanie Stanu Końcowego ---
    final_positions = local_simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(local_simulation.topology, final_positions, open(final_state_path, 'w'))
    report_file.close()
    print(f"--- Simulation Run Completed: {sim_output_dir} ---")

# --- Function for Analysis and Plotting ---
def analyze_trajectory_and_plot_histograms(xtc_file_path, pdb_topology_file_path, output_prefix):
    print(f"\n--- Analyzing Trajectory: {xtc_file_path} ---")
    try:
        traj = md.load(xtc_file_path, top=pdb_topology_file_path) 
        print(f"Loaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms.")

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
        total_steps_production=50000  # Utrzymujemy 50000 * 100
    )

    # --- 10. Analiza trajektorii ---
    analyze_trajectory_and_plot_histograms(
        xtc_file_path=os.path.join(paths['output_dir'], 'trajectory.xtc'),
        pdb_topology_file_path=os.path.join(paths['output_dir'], 'trajectory.pdb'), # Używamy PDB z tego samego przebiegu dla topologii
        output_prefix=os.path.join(paths['output_dir'], 'analysis')
    )

    print(f'\nCompleted processing structure {structure_id}')
    print(f"{'='*50}\n")

print('\nAll simulations completed!')