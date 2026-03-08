# TP4: Low-Budget Learning

Classifying aerial images into two French departments (87 — Haute-Vienne vs. 92 — Hauts-de-Seine) using **very little training data** (42 images) and a **limited compute budget**.

---

## Results

| Question | Method | Val Accuracy | Trained Params |
| --- | --- | --- | --- |
| Q3 | Last layer (random model) | ~50% | 1,026 |
| Q4 | Last layer (ImageNet pretrained model) | **83.3%** | 1,026 |
| Q5 | LoRA (r=4) on pretrained backbone | **90.5%** | 62,542 |
| Q6 | LoRA + Data Augmentation + CutMix | **88.1%** | 62,542 |

---

## Architecture

* **Backbone:** ResNet-10 (ResNet-18 with only 1 block per layer)
* **Final layer:** `LastLayer` — Linear(512, 2)
* **LoRA:** rank r=4, alpha=8, applied to conv1/conv2 of each BasicBlock
* **Augmentations:** RandomResizedCrop, HorizontalFlip, ColorJitter, RandomGrayscale, CutMix (p=0.3)

---

## Reproduction

```bash
# Create the environment
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchmetrics matplotlib peft

# Launch the notebook
jupyter notebook TP4.ipynb
# Or run all cells from VS Code
