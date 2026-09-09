"""
Evaluation of initialized and SSL-pretrained models.

Models are initialized first and pretrained weights are loaded
from the corresponding checkpoints.

Evaluation includes:
    to be defined...
"""


from .utils import initialize_siglip2, initialize_dataloader, load_trained_model
from .dino import train_dino
from .lejepa import train_lejepa


models_to_evaluate = [
    initialize_siglip2("siglip2"),
    initialize_siglip2("imagenet"),

    load_trained_model(
        "siglip2",
        "checkpoints/dino_siglip2.pth"
    ),

    load_trained_model(
        "imagenet",
        "checkpoints/dino_imagenet.pth"
    ),

    # load_trained_model(
    #     "siglip2",
    #     "checkpoints/lejepa_siglip2.pth"
    # ),

    # load_trained_model(
    #     "imagenet",
    #     "checkpoints/lejepa_imagenet.pth"
    # ),
]

def model_evaluation(model, dataloader):
    # 1. Get representations from the model
    # 2. Evaluate representations
    # 3. Return evaluation results
    ...

    return results

