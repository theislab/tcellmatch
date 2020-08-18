# TcellMatch: Predicting T-cell to epitope specificity.

TcellMatch is a collection of models to predict antigen specificity of single T cells based on CDR3 sequences and other 
single cell modalities, such as RNA counts and surface protein counts. 
As labeled training data, either defined CDR3-antigen pairs from data bases such as IEDB or VDJdb are used, or 
pMHC multimer "stained" single cells in which pMHC binding is used as a label indicating specificity to the loaded 
peptide.

This package can be used to train such models and contains the model code, data loaders and 
grid search helper functions.
Accompanying paper: https://www.embopress.org/doi/full/10.15252/msb.20199416

# Installation
This package can be locally installed via pip by first cloning a copy of this repository into you local directory
and then running `pip install -e .` at the top level of this clone's directory.
