from src.models.image_encoder import *
from src.models.pl_modules import *
from src.models.xtransformer import *


def get_model(cfg, **kwargs):
    """
    Initializes models for training purposes

    """

    if "transformer_xformer" in cfg.model.keys():
        encoder, decoder = init_models(cfg)  #

    return encoder, decoder
