# Endoscopy SSL

## Directory Structure

```text
ENDOSCOPY/
├── README.md
├── utils.py          # Model initialization and DataLoader preparation
├── dino.py           # DINOv1 self-supervised training
├── lejepa.py         # LeJEPA self-supervised training
├── evaluation.py     # Model representation evaluation (e.g. k-NN, linear probing, etc.)
└── train.py          # Example usage of the complete pipeline