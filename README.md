# MultiOE-4-GDA-Prediction

This repository is for the publication:"Human-aligned multi-ontologies representation for gene-disease associations prediction."

## Code structure and usage
- **dataset** folder include our processed dataset of TBGA and DisGeNet.

For the original dataset, please check:
TBGA: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04646-6
DisGeNet: https://www.disgenet.org/

Furthermore, the code for preprocessing these two datasets is in processTBGA.py and DG_Miner_preprocess.py, respectively. TBGADataset.py is the code to prepare for the PyTorch dataloader-fashion input.

- TrainBERT.py and TrainERNIE.py include the code for the main training code
- For the code of ERNIE model, please check the original repo: https://github.com/thunlp/ERNIE and put "knowledge_bert" folder into this working directory.
- For the ontology embedding, please check the original repo from OWL2Vec*: https://github.com/KRR-Oxford/OWL2Vec-Star. Create a "embeddings" folder and put output.embeddings file in it.

## Cite
if you find our research useful, please cite: 


