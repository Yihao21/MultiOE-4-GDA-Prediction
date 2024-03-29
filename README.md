# MultiOE-4-GDA-Prediction

This repository is for the publication:"Multi-ontology embeddings approach on human-aligned multi-ontologies representation for gene-disease associations prediction." from Fraunhofer SCAI Applied Semantics group.

Paper link: https://doi.org/10.1016/j.heliyon.2023.e21502

## Requirements
- Pytorch >=0.4.1
- Python3
- owlready2
- tqdm
- spacy
- wandb
- transformers
- gensim
- pandas

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
If you find this code or the paper to be useful for your research, please consider citing.

<pre>
@article{WANG2023e21502,
title = {Multi-ontology embeddings approach on human-aligned multi-ontologies representation for gene-disease associations prediction},
journal = {Heliyon},
pages = {e21502},
year = {2023},
issn = {2405-8440},
doi = {https://doi.org/10.1016/j.heliyon.2023.e21502},
url = {https://www.sciencedirect.com/science/article/pii/S2405844023087108},
author = {Yihao Wang and Philipp Wegner and Daniel Domingo-Fernández and Alpha {Tom Kodamullil}},
keywords = {Multi-ontology, Natural language processing}}</pre>
