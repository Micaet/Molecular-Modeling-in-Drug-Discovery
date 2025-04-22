#!/usr/bin/env zsh

# Configuration
MAX_LIGAND=22  # Set to your max ligand number
OUTPUT="vinardo_scores.csv"

# Write CSV header
echo "ligand_number,vinardo_score" > $OUTPUT

# Loop through natural numbers
for i in {1..$MAX_LIGAND}; do
    file="./docked_ligands/docked_ligand_$i.pdbqt"
    
    if [[ -f "$file" ]]; then
        # Extract the best score (first occurrence)
        score=$(grep "REMARK VINA RESULT" "$file" | head -n 1 | awk '{print $4}')
        echo "$i,$score" >> $OUTPUT
    else
        echo "Warning: $file not found, skipping"
    fi
done

echo "Results saved to $OUTPUT"