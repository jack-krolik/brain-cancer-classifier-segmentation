from dotenv import load_dotenv
import os
import wandb

from src.utils.config import TrainingConfig


def create_wandb_config(config: TrainingConfig, additional_params: dict = {}):
    """
    Create a dictionary from the TrainingConfig object to be used for logging to wandb

    Args:
    - config (TrainingConfig): the training configuration

    Returns:
    - dict: a dictionary containing the training configuration
    """
    return {
        "architecture": config.architecture,
        "dataset": config.dataset,
        **config.hyperparameters.to_dict(),
        **additional_params        
    }

def verify_wandb_config(wb_config: dict, training_config: TrainingConfig):
    """
    Verify that the config dictionary contains all the necessary keys

    Args:
    - wb_config (dict): the dictionary to verify
    - training_config (TrainingConfig): the training configuration

    Returns:
    - bool: True if the dictionary contains all the necessary keys, False otherwise
    """
    hyperparameters = training_config.hyperparameters.to_dict()
    dataset = training_config.dataset
    architecture = training_config.architecture

    match_hyperparameters = all([hyperparameters[key] == wb_config.get(key, None) for key in hyperparameters])
    match_dataset = dataset == wb_config.get("dataset", None)
    match_architecture = architecture == wb_config.get("architecture", None)

    return match_hyperparameters and match_dataset and match_architecture