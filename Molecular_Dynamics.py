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
import itertools

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
        'output_dir': base_output
    }
plt.plot()

# --- Function to run simulation ---
def run_simulation_logic(sim_output_dir, sim_integrator, sim_topology, sim_positions, sim_system, include_heating, total_steps_production):
    print(f"--- Starting Simulation Run: Output to {sim_output_dir} ---")
    
    local_simulation = app.Simulation(sim_topology, sim_system, sim_integrator)
    local_simulation.context.setPositions(sim_positions)

    print(f"Minimizing energy for run: {sim_output_dir}...")
    local_simulation.minimizeEnergy()
    positions_after_local_minimization = local_simulation.context.getState(getPositions=True).getPositions()

    # --- Konfiguracja Reporterów ---
    pdb_reporter_path = os.path.join(sim_output_dir, 'trajectory.pdb')
    xtc_reporter_path = os.path.join(sim_output_dir, 'trajectory.xtc')
    data_reporter_path = os.path.join(sim_output_dir, 'simulation_data.txt')
    # final_state_path = os.path.join(sim_output_dir, 'final_state.pdb') # Zakomentowane

    # --- Bezpośredni zapis klatki początkowej PDB ---
    print(f"Writing initial PDB frame (after local minimization) to: {pdb_reporter_path}")
    with open(pdb_reporter_path, 'w') as f_pdb:
        app.PDBFile.writeFile(sim_topology, positions_after_local_minimization, f_pdb)
    local_simulation.reporters.append(XTCReporter(xtc_reporter_path, 1000))
    
    report_file = open(data_reporter_path, 'w')
    local_simulation.reporters.append(app.StateDataReporter(report_file, 1000, step=True, potentialEnergy=True, temperature=True))
    
    actual_heating_duration_steps = 0
    if include_heating:
        actual_heating_duration_steps = 25000 

    effective_total_steps = total_steps_production
    if include_heating:
        effective_total_steps += actual_heating_duration_steps 

    local_simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, progress=True, remainingTime=True, speed=True, totalSteps=effective_total_steps))

    if include_heating:
        print('Starting heating phase...')
        initial_temp = 50*unit.kelvin
        final_temp = 300*unit.kelvin
        heating_steps_val = 25000 
        n_stages = 250 
        
        if n_stages <= 0: 
            steps_per_stage = heating_steps_val 
            if n_stages < 0: n_stages = 0
        else:
            steps_per_stage = heating_steps_val // n_stages

        print(f"Heating from {initial_temp} to {final_temp} over {heating_steps_val} steps, in {n_stages} stages of {steps_per_stage} steps each.")

        for i in range(n_stages):
            current_stage_fraction = (i + 1) / n_stages
            temp_k_val = initial_temp.value_in_unit(unit.kelvin) + \
                         (final_temp.value_in_unit(unit.kelvin) - initial_temp.value_in_unit(unit.kelvin)) * current_stage_fraction
            temp = temp_k_val * unit.kelvin
            
            sim_integrator.setTemperature(temp)
            local_simulation.step(steps_per_stage)
            
        sim_integrator.setTemperature(final_temp) 
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
    protein_omm_topology_for_vac_calc,
    protein_omm_positions_for_vac_calc,
    ligand_openff_molecule_for_vac_calc,
    ligand_omm_topology_for_vac_calc    
):
    print(f"\n--- Analyzing Trajectory: {xtc_file_path} ---")
    try:
        traj = md.load(xtc_file_path, top=pdb_topology_file_path) 
        print(f"Loaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms.")

        if traj.n_frames == 0:
            print(f"Warning: Trajectory file {xtc_file_path} contains 0 frames. Skipping all trajectory analysis for this run.")
            plt.close('all') 
            return

        # --- Atom Selections ---
        protein_ca_indices = traj.topology.select('protein and name CA')
        protein_indices = traj.topology.select('protein') # All protein atoms (not just CA)

        # Determine ligand resname from provided OpenMM topology (for consistency with vacuum calcs)
        # This logic should align with how ligand_indices_traj is selected for interaction energy.
        ligand_resname_in_traj = "LIG" # Default
        if ligand_omm_topology_for_vac_calc and ligand_omm_topology_for_vac_calc.getNumResidues() > 0:
            first_lig_res = list(ligand_omm_topology_for_vac_calc.residues())[0]
            if first_lig_res.name and first_lig_res.name.strip():
                ligand_resname_in_traj = first_lig_res.name.strip()
        
        ligand_selection_string = f'(resname "{ligand_resname_in_traj}") and (not water)'
        ligand_indices_traj = traj.topology.select(ligand_selection_string)

        if ligand_indices_traj.size == 0 and ligand_openff_molecule_for_vac_calc and ligand_openff_molecule_for_vac_calc.name:
            mol_name = ligand_openff_molecule_for_vac_calc.name.strip()
            if mol_name:
                ligand_selection_string_fallback = f'(resname "{mol_name}") and (not water)'
                ligand_indices_traj_fallback = traj.topology.select(ligand_selection_string_fallback)
                if ligand_indices_traj_fallback.size > 0:
                    ligand_indices_traj = ligand_indices_traj_fallback
                    ligand_resname_in_traj = mol_name # Update if fallback successful
                    print(f"Ligand selected using fallback name '{mol_name}'.")
                else:
                    print(f"Warning: Ligand selection fallback with name '{mol_name}' also failed.")
        
        if ligand_indices_traj.size == 0:
            print(f"CRITICAL WARNING: Could not select ligand atoms using resname '{ligand_resname_in_traj}' (and potential fallbacks). Ligand-specific analyses will be skipped.")
        else:
            print(f"Ligand atoms selected ({ligand_indices_traj.size}) using resname '{ligand_resname_in_traj}'.")

        # --- Ligand-Specific Analyses (New RMSD, New SASA) ---
        if ligand_indices_traj.size > 0:
            print("\n--- Calculating Ligand RMSD ---")
            try:
                traj_ligand_view = traj.atom_slice(ligand_indices_traj)
                traj_ligand_view.superpose(traj_ligand_view[0])
                rmsd_ligand = md.rmsd(traj_ligand_view, traj_ligand_view, frame=0) * 10 # Angstroms
                
                plt.figure()
                plt.hist(rmsd_ligand, bins=50, density=True)
                plt.xlabel('Ligand RMSD (Å)')
                plt.ylabel('Frequency')
                plt.title(f'Ligand RMSD (superposed on ligand)\\\n{os.path.basename(output_prefix)}')
                plt.savefig(f"{output_prefix}_ligand_rmsd_histogram.png")
                plt.clf()
                print(f"Saved Ligand RMSD histogram to {output_prefix}_ligand_rmsd_histogram.png")

                # Add Ligand RMSD vs. Time plot
                time_points_ps = traj.time if hasattr(traj, 'time') and traj.time.size == traj.n_frames else np.arange(traj.n_frames)
                xlabel_time = 'Time (ps)' if hasattr(traj, 'time') and traj.time.size == traj.n_frames else 'Frame Index'
                plt.figure()
                plt.plot(time_points_ps, rmsd_ligand, linestyle='-')
                plt.xlabel(xlabel_time)
                plt.ylabel('Ligand RMSD (Å)')
                plt.title(f'Ligand RMSD vs. Time\\\n{os.path.basename(output_prefix)}')
                plt.grid(True)
                plt.savefig(f"{output_prefix}_ligand_rmsd_timeseries.png")
                plt.clf()
                print(f"Saved Ligand RMSD vs. Time plot to {output_prefix}_ligand_rmsd_timeseries.png")

            except Exception as e:
                print(f"Error calculating Ligand RMSD: {e}")

            print("\n--- Calculating Ligand + Pocket RMSD ---")
            try:
                # Define pocket: protein residues with any atom within 0.5 nm (5 Angstroms) of any ligand atom in traj[0]
                pocket_cutoff = 0.5  # nm
                pocket_protein_atom_indices = set()
                
                protein_residues_for_pocket = [res for res in traj.topology.residues if res.is_protein]

                for res in protein_residues_for_pocket:
                    res_atom_indices = np.array([atom.index for atom in res.atoms])
                    if res_atom_indices.size == 0: continue

                    atom_pairs_for_distance_calc = np.array(list(itertools.product(res_atom_indices, ligand_indices_traj)))
                    if atom_pairs_for_distance_calc.shape[0] == 0: continue 

                    min_dist_to_lig = np.min(md.compute_distances(traj[0], atom_pairs_for_distance_calc))
                    if min_dist_to_lig < pocket_cutoff:
                        for atom in res.atoms:
                            pocket_protein_atom_indices.add(atom.index)
                
                pocket_protein_atoms_np = np.array(sorted(list(pocket_protein_atom_indices)))

                if pocket_protein_atoms_np.size > 0:
                    ligand_plus_pocket_indices = np.unique(np.concatenate((ligand_indices_traj, pocket_protein_atoms_np)))
                    
                    traj_lig_pocket_view = traj.atom_slice(ligand_plus_pocket_indices)
                    traj_lig_pocket_view.superpose(traj_lig_pocket_view[0])
                    rmsd_lig_pocket = md.rmsd(traj_lig_pocket_view, traj_lig_pocket_view, frame=0) * 10 # Angstroms
                    
                    plt.figure()
                    plt.hist(rmsd_lig_pocket, bins=50, density=True)
                    plt.xlabel('Ligand + Pocket RMSD (Å)')
                    plt.ylabel('Frequency')
                    plt.title(f'Ligand + Pocket (Prot atoms < {pocket_cutoff*10:.0f}Å of Lig) RMSD\\\n{os.path.basename(output_prefix)}')
                    plt.savefig(f"{output_prefix}_ligand_pocket_rmsd_histogram.png")
                    plt.clf()
                    print(f"Saved Ligand + Pocket RMSD histogram to {output_prefix}_ligand_pocket_rmsd_histogram.png")

                    time_points_ps_pocket = traj_lig_pocket_view.time if hasattr(traj_lig_pocket_view, 'time') and traj_lig_pocket_view.time.size == traj_lig_pocket_view.n_frames else np.arange(traj_lig_pocket_view.n_frames)
                    xlabel_time_pocket = 'Time (ps)' if hasattr(traj_lig_pocket_view, 'time') and traj_lig_pocket_view.time.size == traj_lig_pocket_view.n_frames else 'Frame Index'

                    plt.figure()
                    plt.plot(time_points_ps_pocket, rmsd_lig_pocket, linestyle='-')
                    plt.xlabel(xlabel_time_pocket)
                    plt.ylabel('Ligand + Pocket RMSD (Å)')
                    plt.title(f'Ligand + Pocket RMSD vs. Time\\\n{os.path.basename(output_prefix)}')
                    plt.grid(True)
                    plt.savefig(f"{output_prefix}_ligand_pocket_rmsd_timeseries.png")
                    plt.clf()
                    print(f"Saved Ligand + Pocket RMSD vs. Time plot to {output_prefix}_ligand_pocket_rmsd_timeseries.png")
                else:
                    print("No protein pocket atoms found within cutoff for Ligand + Pocket RMSD.")
            except Exception as e:
                print(f"Error calculating Ligand + Pocket RMSD: {e}")


            print("\n--- Calculating Ligand SASA Occlusion by Protein ---")
            try:
                sasa_occlusion_percent = []
                sasa_time_points_ps = [] # Time in picoseconds

                num_sasa_frames = min(100, traj.n_frames) # Analyze up to 100 frames
                if num_sasa_frames == 0 and traj.n_frames > 0: num_sasa_frames = 1
                
                sasa_frame_indices = np.linspace(0, traj.n_frames - 1, num_sasa_frames, dtype=int) if num_sasa_frames > 0 else []

                if not sasa_frame_indices.size > 0:
                    print("No frames selected for SASA calculation.")
                else:
                    print(f"Calculating SASA for {len(sasa_frame_indices)} selected frames...")
                    
                    traj_subset_for_sasa_calc = traj[sasa_frame_indices]
                    sasa_all_atoms_for_subset_frames = md.shrake_rupley(traj_subset_for_sasa_calc, mode='atom')

                    for i, original_frame_idx in enumerate(sasa_frame_indices):
                        current_frame_traj_for_isolated_lig_calc = traj[original_frame_idx]
                        ligand_only_traj_frame = current_frame_traj_for_isolated_lig_calc.atom_slice(ligand_indices_traj)
                        
                        if ligand_only_traj_frame.n_atoms == 0: 
                            continue
                        
                        sasa_lig_isolated_atoms_results = md.shrake_rupley(ligand_only_traj_frame, mode='atom')
                        total_sasa_lig_isolated = np.sum(sasa_lig_isolated_atoms_results[0]) 

                        if total_sasa_lig_isolated <= 1e-6: 
                            continue

                        per_atom_sasa_for_this_complex_frame = sasa_all_atoms_for_subset_frames[i] 
                        total_sasa_lig_in_complex = np.sum(per_atom_sasa_for_this_complex_frame[ligand_indices_traj])
                        
                        occlusion = (total_sasa_lig_isolated - total_sasa_lig_in_complex) / total_sasa_lig_isolated * 100
                        
                        sasa_occlusion_percent.append(occlusion)
                        sasa_time_points_ps.append(current_frame_traj_for_isolated_lig_calc.time[0] if current_frame_traj_for_isolated_lig_calc.time.size > 0 else original_frame_idx) 

                if sasa_occlusion_percent:
                    plt.figure()
                    plt.plot(sasa_time_points_ps, sasa_occlusion_percent, marker='o', linestyle='-')
                    plt.xlabel('Time (ps)' if (sasa_time_points_ps and len(sasa_time_points_ps) > 0 and isinstance(sasa_time_points_ps[0], (float, int)) and sasa_time_points_ps[0] != (sasa_frame_indices[0] if sasa_frame_indices.size > 0 else -1) ) else 'Frame Index')
                    plt.ylabel('Ligand SASA Occlusion by Protein (%)')
                    plt.title(f'Ligand SASA Occlusion Over Time\\\n{os.path.basename(output_prefix)}')
                    plt.ylim(min(0, np.nanmin(sasa_occlusion_percent) if sasa_occlusion_percent else 0) - 5, max(100, np.nanmax(sasa_occlusion_percent) if sasa_occlusion_percent else 100) + 5)
                    plt.grid(True)
                    plt.savefig(f"{output_prefix}_ligand_sasa_occlusion_plot.png")
                    plt.clf()
                    print(f"Saved Ligand SASA Occlusion plot to {output_prefix}_ligand_sasa_occlusion_plot.png")
                else:
                    print("No data points for Ligand SASA Occlusion plot.")
            except Exception as e:
                print(f"Error calculating Ligand SASA Occlusion: {e}")
                import traceback
                traceback.print_exc()
        # End of ligand-specific analyses

        if protein_ca_indices.size == 0:
            print("Warning: Could not find protein C-alpha atoms. Skipping RMSD and RMSF analysis.")
        else:
            print("\nCalculating Protein C-alpha RMSD...")
            rmsd = md.rmsd(traj, traj, frame=0, atom_indices=protein_ca_indices)
            plt.figure()
            plt.hist(rmsd * 10, bins=50, density=True) # Angstroms
            plt.xlabel('Protein C-alpha RMSD (Å)')
            plt.ylabel('Frequency')
            plt.title(f'Protein C-alpha RMSD (vs. first frame)\\\n{os.path.basename(output_prefix)}')
            plt.savefig(f"{output_prefix}_protein_ca_rmsd_histogram.png")
            plt.clf()
            print(f"Saved Protein C-alpha RMSD histogram to {output_prefix}_protein_ca_rmsd_histogram.png")

            if protein_ca_indices.size > 0:
                print("\nCalculating Protein C-alpha RMSF...")
                ref_ca_indices = traj.topology.select('protein and name CA and backbone') 
                if ref_ca_indices.size == 0: ref_ca_indices = protein_ca_indices 
                
                if ref_ca_indices.size > 0:
                    traj.superpose(traj[0], atom_indices=ref_ca_indices) 
                    rmsf_per_atom = md.rmsf(traj, traj[0], atom_indices=protein_ca_indices) 
                    rmsf_values_A = rmsf_per_atom * 10 

                    ca_atom_indices_for_plot = np.arange(len(rmsf_values_A))
                    ca_indices_near_ligand_mask = np.zeros(len(protein_ca_indices), dtype=bool)

                    if ligand_indices_traj.size > 0:
                        print("Identifying C-alpha atoms near the ligand for RMSF plot highlighting...")
                        proximity_cutoff_A = 4.0 
                        proximity_cutoff_nm = proximity_cutoff_A / 10.0
                        original_traj_frame0 = md.load_frame(xtc_file_path, index=0, top=pdb_topology_file_path)

                        for i, ca_atom_global_idx in enumerate(protein_ca_indices):
                            if ligand_indices_traj.size > 0: 
                                distances_ca_to_lig = md.compute_distances(original_traj_frame0, atom_pairs=np.array([[ca_atom_global_idx, lig_idx] for lig_idx in ligand_indices_traj]))
                                if distances_ca_to_lig.size > 0 and np.min(distances_ca_to_lig) < proximity_cutoff_nm:
                                    ca_indices_near_ligand_mask[i] = True
                    
                    plt.figure(figsize=(10,6))
                    plt.plot(ca_atom_indices_for_plot[~ca_indices_near_ligand_mask], 
                             rmsf_values_A[~ca_indices_near_ligand_mask], 
                             marker='.', linestyle='-', color='blue', label='Cα (Distant from Ligand)')
                    if np.any(ca_indices_near_ligand_mask):
                        plt.plot(ca_atom_indices_for_plot[ca_indices_near_ligand_mask], 
                                 rmsf_values_A[ca_indices_near_ligand_mask], 
                                 marker='o', linestyle='-', color='red', markersize=5, label=f'Cα (Near Ligand, < {proximity_cutoff_A}Å)')
                    
                    plt.xlabel('C-alpha Atom Index (in selection order)')
                    plt.ylabel('RMSF (Å)')
                    plt.title(f'Protein C-alpha RMSF (superposed on Cα backbone)\\\n{os.path.basename(output_prefix)}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{output_prefix}_protein_ca_rmsf_plot.png")
                    plt.clf()
                    print(f"Saved Protein C-alpha RMSF plot to {output_prefix}_protein_ca_rmsf_plot.png")
                else:
                    print("Skipping RMSF because no C-alpha atoms for reference superposition were found.")
        

        # --- Calculate MM Interaction Energy (Simplified Delta G component) ---
        print("\n--- Calculating MM Interaction Energy ---")
        try:
            protein_ff = app.ForceField('amber14-all.xml')
            protein_vac_system = protein_ff.createSystem(protein_omm_topology_for_vac_calc,
                                                         nonbondedMethod=app.NoCutoff,
                                                         constraints=None,
                                                         rigidWater=False)
            ligand_ff_str = 'openff_unconstrained-2.1.0.offxml' 
            ligand_openff_ff_vac = OpenFFForceField(ligand_ff_str, allow_cosmetic_attributes=True)


            ligand_vac_system = ligand_openff_ff_vac.create_openmm_system(
                ligand_openff_molecule_for_vac_calc.to_topology(),
                charge_from_molecules=[ligand_openff_molecule_for_vac_calc],
            )
            for force in ligand_vac_system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                    force.setUseDispersionCorrection(False)
            complex_vac_modeller = app.Modeller(protein_omm_topology_for_vac_calc, protein_omm_positions_for_vac_calc)
            complex_vac_modeller.add(ligand_omm_topology_for_vac_calc, ligand_openff_molecule_for_vac_calc.conformers[0].to_openmm())
            complex_vac_topology = complex_vac_modeller.getTopology()

            complex_system_generator_vac = SystemGenerator(
                forcefields=['amber14-all.xml'], 
                small_molecule_forcefield=ligand_ff_str,
                molecules=[ligand_openff_molecule_for_vac_calc],
                cache='openff_cache.sqlite'
            )
            complex_vac_system = complex_system_generator_vac.create_system(
                complex_vac_topology
            )
            for force in complex_vac_system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                    force.setUseDispersionCorrection(False) 
                    if force.getNonbondedMethod() == mm.NonbondedForce.PME:
                        print("Warning: PME was set on complex_vac_system's NonbondedForce despite NoCutoff request. Attempting to ensure NoCutoff.")
            
            
            num_constraints_complex_vac = complex_vac_system.getNumConstraints()
            if num_constraints_complex_vac > 0:
                print(f"Note: Complex vacuum system initially has {num_constraints_complex_vac} constraints. Removing them for pure potential energy calculation.")
               
                pass 

            ligand_resname_in_traj_for_ie = "LIG" 
            if ligand_omm_topology_for_vac_calc.getNumResidues() > 0:
                first_lig_res_for_ie = list(ligand_omm_topology_for_vac_calc.residues())[0]
                if first_lig_res_for_ie.name:
                    ligand_resname_in_traj_for_ie = first_lig_res_for_ie.name
            
            protein_selection_string_for_ie = f'(protein and not water) and not (resname "{ligand_resname_in_traj_for_ie}")'
            protein_indices_traj_for_ie = traj.topology.select(protein_selection_string_for_ie)
            
            ligand_selection_string_for_ie = f'(resname "{ligand_resname_in_traj_for_ie}") and (not water)'
            ligand_indices_traj_for_ie = traj.topology.select(ligand_selection_string_for_ie)

            if ligand_indices_traj_for_ie.size == 0 and ligand_openff_molecule_for_vac_calc.name:
                 mol_name_for_ie = ligand_openff_molecule_for_vac_calc.name.strip()
                 if mol_name_for_ie:
                    ligand_selection_string_fallback_for_ie = f'(resname "{mol_name_for_ie}") and (not water)'
                    ligand_indices_traj_for_ie = traj.topology.select(ligand_selection_string_fallback_for_ie)
                    if ligand_indices_traj_for_ie.size > 0:
                        ligand_resname_in_traj_for_ie = mol_name_for_ie 
                        protein_selection_string_for_ie = f'(protein and not water) and not (resname "{ligand_resname_in_traj_for_ie}")'
                        protein_indices_traj_for_ie = traj.topology.select(protein_selection_string_for_ie)

            if ligand_indices_traj_for_ie.size == 0: 
                print(f"Critical Warning IE: Could not select ligand by resname '{ligand_resname_in_traj_for_ie}' or molecule name. IE calc will fail.")
                
            err_msg_atom_count = ""
            if not protein_omm_topology_for_vac_calc.getNumAtoms() == protein_indices_traj_for_ie.size:
                err_msg_atom_count += f"Protein atom count mismatch IE: VAC_TOPO={protein_omm_topology_for_vac_calc.getNumAtoms()}, TRAJ_SELECT={protein_indices_traj_for_ie.size}.\n"
            if not ligand_omm_topology_for_vac_calc.getNumAtoms() == ligand_indices_traj_for_ie.size:
                err_msg_atom_count += f"Ligand atom count mismatch IE: VAC_TOPO={ligand_omm_topology_for_vac_calc.getNumAtoms()}, TRAJ_SELECT={ligand_indices_traj_for_ie.size} (Resname used: '{ligand_resname_in_traj_for_ie}').\n"
            if not complex_vac_topology.getNumAtoms() == (protein_indices_traj_for_ie.size + ligand_indices_traj_for_ie.size):
                 err_msg_atom_count += f"Complex atom count mismatch IE: VAC_TOPO={complex_vac_topology.getNumAtoms()}, TRAJ_SUM_SELECT={protein_indices_traj_for_ie.size + ligand_indices_traj_for_ie.size}.\n"

            if err_msg_atom_count:
                print("ERROR IE: Atom count mismatch between trajectory selection and vacuum topologies.")
                print(err_msg_atom_count)
                raise ValueError("Atom count mismatch for interaction energy calculation. Check ligand selection logic.")
                        
            protein_vac_context = mm.Context(protein_vac_system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
            ligand_vac_context = mm.Context(ligand_vac_system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
            complex_vac_context = mm.Context(complex_vac_system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
            
            interaction_energies_kj_mol = []
            
            num_frames_for_ie = min(50, traj.n_frames // 20 if traj.n_frames >= 100 else traj.n_frames) 
            if num_frames_for_ie == 0 and traj.n_frames > 0: num_frames_for_ie = 1 
            
            if num_frames_for_ie > 0:
                frame_indices_for_ie = np.linspace(0, traj.n_frames - 1, num_frames_for_ie, dtype=int)
                selected_traj_for_ie = traj[frame_indices_for_ie]
                print(f"Calculating interaction energies for {selected_traj_for_ie.n_frames} frames...")

                for frame_idx, frame in enumerate(selected_traj_for_ie):
                    frame_xyz_nm = frame.xyz[0]

                    protein_pos_nm = frame_xyz_nm[protein_indices_traj_for_ie]
                    ligand_pos_nm = frame_xyz_nm[ligand_indices_traj_for_ie]
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
                plt.hist(interaction_energies_kj_mol, bins=max(15, len(interaction_energies_kj_mol)//2), density=True) 
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
        
        plt.close('all') # Close any remaining plots

    except Exception as e:
        print(f"Error during analysis of {xtc_file_path}: {e}")
        import traceback # Add traceback for the main analysis function error
        traceback.print_exc() # Print traceback for the main analysis function error
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