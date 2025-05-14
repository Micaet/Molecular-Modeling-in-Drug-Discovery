import openmm as mm
from openmm import app
from openmm import unit
from sys import stdout
from pdbfixer import PDBFixer
import xml.etree.ElementTree as ET
# --- Główny skrypt ---
# 2. Naprawa białka
fixer = PDBFixer('dynamics/66/input/5cno_prepared.pdb')
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.removeHeterogens(False)
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)
app.PDBFile.writeFile(fixer.topology, fixer.positions, open('dynamics/66/input/protein_fixed.pdb', 'w'))

# 3. Wczytanie struktur
protein = app.PDBFile('dynamics/66/input/protein_fixed.pdb')
ligand = app.PDBFile('dynamics/66/input/ligand66_pdb.pdb')

# 4. Połączenie systemu
modeller = app.Modeller(protein.topology, protein.positions)
modeller.add(ligand.topology, ligand.positions)

# 5. Sprawdzenie nierozpoznanych reszt
print("\nSprawdzanie nierozpoznanych reszt...")
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml', 'dynamics/66/input/unl.xml')

# Teraz możemy bezpiecznie przejść dalej z forcefield, który ma wszystkie potrzebne szablony

'''
# --- 5. Diagnostyka liganda ---
print("\nAnaliza struktury liganda UNL...")
unl_atoms = []
unl_bonds = []
for res in modeller.topology.residues():
    if res.name == 'UNL':
        print(f"\nZnaleziono ligand UNL (ID: {res.id}):")
        print("Atomy w ligandzie:")
        for atom in res.atoms():
            print(f"  - {atom.name} (typ: {atom.element.symbol})")
            unl_atoms.append((atom.name, atom.element.symbol))
        print("\nWiązania w ligandzie:")
        for bond in res.bonds():
            print(f"  - {bond[0].name} - {bond[1].name}")
            unl_bonds.append((bond[0].name, bond[1].name))
        break
'''

# --- 6. Usuń wszystkie atomy wodoru z białka ---
print("\nUsuwanie wszystkich atomów wodoru z białka...")
atoms_to_delete = []
for atom in modeller.topology.atoms():
    if atom.element.symbol == 'H' and atom.residue.name != 'UNL':
        atoms_to_delete.append(atom)
modeller.delete(atoms_to_delete)
print(f"Usunięto {len(atoms_to_delete)} atomów wodoru\n")

# --- 7. Dodaj standardowe atomy wodoru do białka ---
print("Dodawanie standardowych atomów wodoru do białka...")
modeller.addHydrogens(forcefield=forcefield, pH=7.0)
print("Atomy wodoru dodane\n")

# --- 8. Diagnostyka ASN ---
"""
print("\n=== SZCZEGÓŁOWA INFORMACJA O RESZTACH ASN ===")
asn_count = 0
for res in modeller.topology.residues():
    if res.name == 'ASN':
        asn_count += 1
        print(f"\nASN #{asn_count}:")
        print(f"  Chain: {res.chain.id}")
        print(f"  Residue ID: {res.id}")
        print(f"  Residue Index: {res.index + 1}")
        print("  Atomy w reszcie:")
        for atom in res.atoms():
            print(f"    - {atom.name} (typ: {atom.element.symbol})")
print(f"\nZnaleziono {asn_count} reszt ASN w strukturze")
print("=== KONIEC INFORMACJI O ASN ===\n")
"""

# --- 9. Solwatacja i dalszy workflow ---
modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)

system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)

temperature = 300*unit.kelvin
friction = 1.0/unit.picosecond
timestep = 2.0*unit.femtoseconds
integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)

simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

print('Minimizing energy...')
simulation.minimizeEnergy()

simulation.reporters.append(app.PDBReporter('dynamics/66/output/trajectory.pdb', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))

print('Running simulation...')
simulation.step(50000)

positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(simulation.topology, positions, open('dynamics/66/output/final_state.pdb', 'w'))

print('Simulation completed!')
