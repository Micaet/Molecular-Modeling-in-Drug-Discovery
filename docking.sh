#!/usr/bin/env zsh

# Configuration
RECEPTOR="5cno_clean.pdbqt"
PREPROCESSED_DIR="preprocessed_ligands"
DOCKED_DIR="docked_ligands"
VINA_EXEC="./vina"

# Docking parameters
CENTER_X=-48.405
CENTER_Y=-37.514
CENTER_Z=19.017
SIZE_X=20
SIZE_Y=20
SIZE_Z=20
SCORING="vinardo"
EXHAUSTIVENESS=24

# Create output directory
mkdir -p "$DOCKED_DIR"

# Process ligands 1 through 22
for i in {1..22}; do
    ligand="$PREPROCESSED_DIR/ligand_$i.pdbqt"
    output="$DOCKED_DIR/docked_ligand_$i.pdbqt"

    echo "Docking ligand $i..."
    
    if [[ ! -f "$ligand" ]]; then
        echo "Warning: $ligand not found, skipping"
        continue
    fi

    "$VINA_EXEC" \
        --receptor "$RECEPTOR" \
        --ligand "$ligand" \
        --center_x $CENTER_X --center_y $CENTER_Y --center_z $CENTER_Z \
        --size_x $SIZE_X --size_y $SIZE_Y --size_z $SIZE_Z \
        --scoring $SCORING \
        --exhaustiveness $EXHAUSTIVENESS \
        --out "$output"

    if [[ $? -eq 0 ]]; then
        echo "Success: $output"
    else
        echo "Error docking ligand $i"
    fi
done

echo "Docking complete! Results saved to $DOCKED_DIR/"