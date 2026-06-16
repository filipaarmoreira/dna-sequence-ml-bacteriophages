# Leveraging DNA Sequence Representations to Enhance Machine Learning Models: A Case Study on Bacteriophages

This repository contains the code developed for the benchmarking study of DNA sequence representation strategies for predicting bacteriophage lifecycle.

## Project Overview

Bacteriophages can follow two life cycles: **lytic** (host cell destruction) and **lysogenic** (viral genome integration into the host). Predicting which cycle a phage will follow has important implications for phage therapy and microbiology research.

This project benchmarks four DNA sequence representation strategies combined with four machine learning classifiers, evaluated on a dataset of 1669 phage sequences.

## Dataset

- **1669 phage sequences**: 1152 lytic, 517 lysogenic
- Sources: BACPHLIP, Deephage, PhagePred dataset containing literature-based lytic and lysogenic phage annotations
- Sequences retrieved from NCBI, duplicate and null sequences removed, followed by redundancy reduction using CD-HIT at 80% sequence identity

## Representation Strategies

| Strategy | Description |
|---|---|
| Seq2Vec | 4-mer CountVectorizer  |
| HyenaDNA | Pre-trained DNA foundation model (large-1m) |
| DNABERT-2 | Pooling of DNABERT-2 embeddings |
| Nucleotide Transformer | NT v2-500M multi-species embeddings |

## Classifiers

- Random Forest
- XGBoost
- SVM
- DNN

## Evaluation Protocol

- 5×5 Repeated Stratified K-Fold Cross-Validation, F1 macro as primary metric
- Statistical comparison: Shapiro-Wilk normality test → Friedman test → Conover-Friedman post-hoc with Holm correction
- Fine-tuning (HyenaDNA)

## Results Summary

| Phase | Approach | Best F1 Macro |
|---|---|---|
| 1 | Pre-trained embeddings + classifiers | ~0.90 |
| 2 | HyenaDNA fine-tuning (32k bp) | ~0.69 |

## Repository Structure

```
├── models/          # Embedding extraction scripts
│   ├── seq2vec.py
│   ├── hyenadna.py
│   ├── dnabert2.py
│   └── nucleotide_transformer.py
├── evaluation/      # Classification and statistical analysis scripts
│   ├── cv_classifiers.py
│   ├── qq_plot_normality.py 
│   ├── n_parametric.py
│   ├── heatmap.py
│   └── confusion_matrix.py
└── finetuning/      # Fine-tuning script with HyenaDNA
    └── ft.py
```

## Requirements

See `requirements.txt` for the full list of dependencies.

```bash
pip install -r requirements.txt
```

## Resources

- [Dataset and Article](https://drive.google.com/drive/folders/1E6qwYq27A8e8bfdHqbZgEefG33YjYwMJ?usp=drive_link) 
