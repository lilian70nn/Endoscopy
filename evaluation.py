from .utils import initialize_siglip2, initialize_dataloader
from .dino import train_dino
from .lejepa import train_lejepa

models_to_evaluate = [
    initialize_siglip2(initialization="siglip2"),
    initialize_siglip2(initialization="imagenet"),
    train_dino(initialize_siglip2(initialization="siglip2"), initialize_dataloader(batch_size=64)),
    train_dino(initialize_siglip2(initialization="imagenet"), initialize_dataloader(batch_size=64)),
    train_lejepa(initialize_siglip2(initialization="siglip2"), initialize_dataloader(batch_size=64)),
    train_lejepa(initialize_siglip2(initialization="imagenet"), initialize_dataloader(batch_size=64))
]

def model_evaluation(model, dataloader):
    # 1. Get representations from the model
    # 2. Evaluate representations, e.g.:
    #    - k-NN
    #    - Linear probing
    #    - Other evaluation methods
    # 3. Return evaluation results
    ...

    return results

