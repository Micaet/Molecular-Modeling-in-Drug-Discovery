import openmm as mm
from openmm import app
from openmm import unit
from sys import stdout
from pdbfixer import PDBFixer
import xml.etree.ElementTree as ET
import os

# Dodaj na początku skryptu, przed rozpoczęciem symulacji
output_dir = 'dynamics/66/output'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Inicjalizacja ---
fixer = PDBFixer('dynamics/66/input/5cno_prepared.pdb')
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.removeHeterogens(False)
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)
app.PDBFile.writeFile(fixer.topology, fixer.positions, open('dynamics/66/input/protein_fixed.pdb', 'w'))

# --- 2. Wczytanie struktur ---
protein = app.PDBFile('dynamics/66/input/protein_fixed.pdb')
ligand = app.PDBFile('dynamics/66/input/ligand66_pdb.pdb')

# --- 3. Połączenie systemu ---
modeller = app.Modeller(protein.topology, protein.positions)
modeller.add(ligand.topology, ligand.positions)

# --- 4. Sprawdzenie nierozpoznanych reszt ---
print("\nChecking for unrecognized residues...")
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml', 'dynamics/66/input/unl.xml')
# --- Diagnostyka Liganda (opcjonalne) ---
'''
# --- 5. Diagnostyka liganda ---
print("\nAnalyzing UNL ligand structure...")
unl_atoms = []
unl_bonds = []
for res in modeller.topology.residues():
    if res.name == 'UNL':
        print(f"\nFound ligand UNL (ID: {res.id}):")
        print("Atoms in ligand:")
        for atom in res.atoms():
            print(f"  - {atom.name} (type: {atom.element.symbol})")
            unl_atoms.append((atom.name, atom.element.symbol))
        print("\nBonds in ligand:")
        for bond in res.bonds():
            print(f"  - {bond[0].name} - {bond[1].name}")
            unl_bonds.append((bond[0].name, bond[1].name))
        break
'''

# --- 5. Modyfikacja Wodorów w Białku ---
print("\nRemoving all hydrogen atoms from the protein...")
atoms_to_delete = []
for atom in modeller.topology.atoms():
    if atom.element.symbol == 'H' and atom.residue.name != 'UNL':
        atoms_to_delete.append(atom)
modeller.delete(atoms_to_delete)
print(f"Removed {len(atoms_to_delete)} hydrogen atoms\n")

# --- 6. Dodawanie Standardowych Wodorów do Białka ---
print("Adding standard hydrogen atoms to the protein...")
modeller.addHydrogens(forcefield=forcefield, pH=7.0)
print("Hydrogen atoms added\n")

# --- 7. Diagnostyka ASN (opcjonalne) ---
"""
print("\n=== DETAILED INFORMATION ABOUT ASN RESIDUES ===")
asn_count = 0
for res in modeller.topology.residues():
    if res.name == 'ASN':
        asn_count += 1
        print(f"\nASN #{asn_count}:")
        print(f"  Chain: {res.chain.id}")
        print(f"  Residue ID: {res.id}")
        print(f"  Residue Index: {res.index + 1}")
        print("  Atoms in residue:")
        for atom in res.atoms():
            print(f"    - {atom.name} (type: {atom.element.symbol})")
print(f"\nFound {asn_count} ASN residues in the structure")
print("=== END OF ASN INFORMATION ===\n")
"""

# --- 9. Solwatacja i Konfiguracja Symulacji ---
modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)

# --- 10. Tworzenie Systemu i Integratora ---
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)

temperature = 300*unit.kelvin
friction = 1.0/unit.picosecond
timestep = 2.0*unit.femtoseconds
integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)

# --- 11. Ustawienie Symulacji ---
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# --- 12. Minimalizacja Energii ---
print('Minimizing energy...')
simulation.minimizeEnergy()

# --- 13. Konfiguracja Reporterów ---
simulation.reporters.append(app.PDBReporter('dynamics/66/output/trajectory.pdb', 1000))
report_file = open('dynamics/66/output/simulation_data.txt', 'w')
# !!! zapisac dynamike (xtc reporter)
# !!! delta G, rmsf, rmsd, sasa (pytraj) (hiustogramy dla nich)
# !!! czas razy 100
# !!! openmmforcefiels openfftoolkit <- xml
simulation.reporters.append(app.StateDataReporter(report_file, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, progress=True, remainingTime=True, speed=True, totalSteps=50000))
# !!! przed symulacja zrobic symulacje rozgrzewkowa gdzie co ilesc timestepow ogrzewamy o 1 K,
# np od 50 do 300
# --- 14. Uruchomienie Symulacji ---
print('Running simulation...')
simulation.step(50000)

# --- 15. Zapisanie Stanu Końcowego ---
positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(simulation.topology, positions, open('dynamics/66/output/final_state.pdb', 'w'))
report_file.close()

# --- 16. Zakończenie ---
print('Simulation completed!')
print(f"Trajectory saved to: dynamics/66/output/trajectory.pdb")
print(f"Final state saved to: dynamics/66/output/final_state.pdb")
print(f"Simulation data saved to: dynamics/66/output/simulation_data.txt")
