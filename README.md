# TP4 — Low-Budget Learning

Classifier des images aériennes en deux départements français (87 — Haute-Vienne vs 92 — Hauts-de-Seine) avec **très peu de données d'entraînement** (42 images) et un **budget compute limité**.


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

