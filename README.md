# Molecular Modeling in Drug Discovery

## Project Overview
This project aims to identify a potent ligand inhibitor for the 5CNO protein (EGFR kinase), which is implicated in various cancers. By leveraging state-of-the-art computational techniques—including molecular docking, molecular dynamics simulations using OpenMM, and binding affinity estimation—we replicate a modern computational drug discovery pipeline.

## Project Stages
The project is structured into four stages, each focusing on a critical aspect of the drug discovery workflow:

1. **Protein and Ligand Preparation** *(Completed)*  
   - Selection and preparation of the EGFR kinase protein (PDB ID: 5CNO)  
   - Ligand library assembly and preprocessing  
   - File: `Molecular-Modeling-In-Drug-Discovery.ipynb`

2. **Virtual Screening via Molecular Docking** *(Completed)*  
   - High-throughput virtual screening using AutoDock Vina  
   - Scoring and ranking of ligand binding poses  
   - Results stored in: `docking/`  
   - File: `Molecular-Modeling-In-Drug-Discovery.ipynb`

3. **Molecular Dynamics Simulation** *(Completed)*  
   - Post-docking refinement through molecular dynamics  
   - Simulations implemented using **OpenMM** to evaluate stability of top ligand–protein complexes  
   - The `dynamics/` folder contains all necessary input files and scripts to run the simulation  
   - **Note:** Output trajectory and log files are not included due to file size; users must run the simulation locally to generate them  
   - File: `Molecular_dynamics.py`

4. **Binding Free Energy and Affinity Analysis** *(In Progress)*  
   - Estimation of binding free energies using MM/PBSA or similar methods  
   - Comparative analysis of top ligands  
   - File: `Molecular-Modeling-In-Drug-Discovery.ipynb`

---

## Dependencies
To reproduce the results, please ensure all packages listed in `requirements.txt` are installed.  
Recommended environment: Python 3.8+ with conda or virtualenv. OpenMM installation may require additional system-specific setup (e.g., GPU drivers, CUDA).

---

## Authors
- Antoni Rakowski  
- Kacper Rzeźniczak  
- Michał Syrkiewicz
