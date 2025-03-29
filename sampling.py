from rdkit import Chem
import random

random.seed(2137)

# Path to SDF file
input_sdf = "Enamine_Kinase_Library_plated_64960cmpds_20250316.sdf"

# Load molecules from the SDF file; skip any molecules that fail to load
mols = [mol for mol in Chem.SDMolSupplier(input_sdf) if mol is not None]

# Number of ligands to sample
n_sample = 10000

if len(mols) < n_sample:
    print(f"Only {len(mols)} ligands available. Selecting all molecules.")
    sampled_mols = mols
else:
    # Randomly sample without replacement
    sampled_mols = random.sample(mols, n_sample)

# Write the selected molecules to a new SDF file
output_sdf = "sampled_ligands.sdf"
writer = Chem.SDWriter(output_sdf)
for mol in sampled_mols:
    writer.write(mol)
writer.close()

print(f"Selected {len(sampled_mols)} ligands have been saved to {output_sdf}.")