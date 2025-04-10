from pathlib import Path

import torch
import torch.nn as nn
import os
# import utils.logging
from . import unet
from . import paths
from .config import ModelConfig

# logger = utils.logging.get_logger(__name__)


def load_model(
    model_name: str,
    source: str | Path,
    load_weights: bool = True,
    key: str = "ema_model",
    return_config: bool = False,
    snapshot_iter: int | None = None,
    from_pretrained: bool = False,
) -> nn.Module | tuple[nn.Module, ModelConfig]:
    """
    Load a model from a given source.

    Parameters:
    ----------
    source : str or Path
        The source of the model config. It can be a file or a directory if Path,
        or a model name if str.
    load_weights : bool, optional
        Whether to load the model weights. Defaults to True.
    key : str, optional
        The dict key to use when loading the model weights. Defaults to "ema_model".
    return_config : bool, optional
        Whether to return the model configuration. Defaults to False.
    snapshot_iter : int, optional
        The iteration number of the snapshot to load. Defaults to None.
    from_pretrained : bool, optional
        Whether to load the model from the PRETRAINED_PARENT. Will be used if
        model name is passed as 'source'. If False, the MODEL_PARENT directory
        is used. Set this to False if you want to use your own trained model.
        Defaults to False.

    Returns:
    ----------
    model : nn.Module
        The loaded model.
    config : modelConfig, optional
        The model configuration, if `return_config` is True.

    Raises:
    ----------
    ValueError
        If 'source' is an invalid model identifier.
    FileNotFoundError
        If the specified snapshot file is not found.

    """

    model_file = source
    config_file = os.path.join(os.path.dirname(__file__), f'models/config_{model_name}.json')

    # Load config and construct model
    # logger.info(f"Loading model from {config_file}")
    config = ModelConfig.from_preset(config_file)
    model = unet.EDMPrecond.from_config(config)

    # Load model weights
    if load_weights:

        # Load snapshot, if specified
        if snapshot_iter is not None:
            snapshot_file = model_dir / f"snapshots/snapshot_iter_{iter:08d}.pt"
            if not snapshot_file.exists():
                raise FileNotFoundError(f"Snapshot file {snapshot_file} not found.")
            # logger.info(f"Loading snapshot from {snapshot_file}")
            model = load_parameters(model, snapshot_file, key=key)

        # Otherwise, load final model
        else:
            model = load_parameters(model, model_file, key=key)

    # Return desired output
    if return_config:
        return model, config
    else:
        return model


def load_parameters(
    model: nn.Module, path: str | Path, key: str = "ema_model"
) -> nn.Module:
    """
    Load model parameters from a file.

    Parameters
    ----------
    model : nn.Module
        Model to load parameters into.
    path : str or Path
        Path to the file containing the model parameters.
    key : str, optional
        The key to load the state_dict from parameters file. Defaults to "ema_model".

    Returns
    -------
    model: nn.Module
        Model with loaded parameters.
    """
    # logger.info(f"Loading model parameters from {path}")

    # Import model weights from file
    state_dict = torch.load(path, map_location="cpu")[key]

    # If necessary, remove 'module.' from keys (e.g. for ema model)
    if key != "model":
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }

    # Load weights into model
    model.load_state_dict(state_dict)

    return model


def isModel(model: nn.Module, modelClass: type) -> bool:
    """
    Check whether a model is an instance of a given class.

    Parameters
    ----------
    model : nn.Module
        The model to check.
    modelClass : type
        The class to check against.

    Returns
    -------
    bool
        True if model is an instance of modelClass, False otherwise.
    """
    if isinstance(model, nn.DataParallel):
        return isinstance(model.module, modelClass)
    return isinstance(model, modelClass)
