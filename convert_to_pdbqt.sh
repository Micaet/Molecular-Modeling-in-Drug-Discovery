#!/usr/bin/env zsh

# Converts PDB to PDBQT format using Open Babel

# Input and output files
INPUT_PDB="5cno_prepared.pdb"
OUTPUT_PDBQT="5cno_clean.pdbqt"

# Check if input file exists
if [[ ! -f "$INPUT_PDB" ]]; then
    echo "❌ Error: Input file $INPUT_PDB not found!" >&2
    exit 1
fi

# Check if obabel is installed
if ! command -v obabel >/dev/null 2>&1; then
    echo "❌ Error: Open Babel (obabel) is not installed!" >&2
    echo "Install it with: conda install -c conda-forge openbabel" >&2
    exit 1
fi

# Run the conversion
echo "Converting $INPUT_PDB to $OUTPUT_PDBQT..."
if obabel "$INPUT_PDB" -O "$OUTPUT_PDBQT" -xr; then
    # Verify output was created
    if [[ -f "$OUTPUT_PDBQT" ]]; then
        echo "Successfully created $OUTPUT_PDBQT"
        exit 0
    else
        echo "❌ Error: Conversion failed - output file not created" >&2
        exit 1
    fi
else
    echo "❌ Error: Conversion failed" >&2
    exit 1
fi