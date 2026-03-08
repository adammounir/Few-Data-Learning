# TP4 — Low-Budget Learning

> MVA — Reconnaissance des formes et méthodes neuronales  
> Session 4 : Transfer Learning, LoRA & Data Augmentation

## Objectif

Classifier des images aériennes en deux départements français (87 — Haute-Vienne vs 92 — Hauts-de-Seine) avec **très peu de données d'entraînement** (42 images) et un **budget compute limité**.

## Structure du projet

```
├── TP4.ipynb                  # Notebook principal (questions + résultats)
├── last_layer.py              # Dernière couche du ResNet adaptée (Q1)
├── utils.py                   # Fonction precompute_features (Q2)
├── last_layer_finetune.pth    # Poids de la dernière couche fine-tunée (Q4)
├── final_model.pth            # Poids du modèle complet LoRA + augmentation (Q6)
├── drawing_lora.png           # Schéma explicatif de LoRA (Q5)
├── cutmix.png                 # Schéma explicatif de CutMix (Q6)
├── data/                      # Données (non versionné)
│   └── TP4_images/
│       ├── north_dataset_sample/   # Train (42 images)
│       └── north_dataset_test/     # Validation (42 images)
└── README.md
```

## Résultats

| Question | Méthode | Val Accuracy | Params entraînés |
|----------|---------|:------------:|:----------------:|
| Q3 | Last layer (modèle random) | ~50% | 1 026 |
| Q4 | Last layer (modèle pretrained ImageNet) | **83.3%** | 1 026 |
| Q5 | LoRA (r=4) sur backbone pretrained | **90.5%** | 62 542 |
| Q6 | LoRA + Data Augmentation + CutMix | **88.1%** | 62 542 |

## Architecture

- **Backbone :** ResNet-10 (ResNet-18 avec 1 seul block par layer)
- **Dernière couche :** `LastLayer` — Linear(512, 2)
- **LoRA :** rang r=4, alpha=8, appliqué sur conv1/conv2 de chaque BasicBlock
- **Augmentations :** RandomResizedCrop, HorizontalFlip, ColorJitter, RandomGrayscale, CutMix (p=0.3)

## Fichiers à soumettre

### Sur le cours (grading)

1. `TP4.ipynb`
2. `utils.py`
3. `drawing_lora.png`
4. `cutmix.png`

### Sur Codabench (évaluation test set)

1. `last_layer.py`
2. `last_layer_finetune.pth`
3. `final_model.pth`

## Reproduction

```bash
# Créer l'environnement
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchmetrics matplotlib peft

# Lancer le notebook
jupyter notebook TP4.ipynb
# Ou exécuter toutes les cellules depuis VS Code
```

## Auteur

Adam Mounir — MVA 2025-2026
