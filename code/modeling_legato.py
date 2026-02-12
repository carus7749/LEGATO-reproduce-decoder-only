import os
import torch
from typing import Optional, Union, List, Tuple
from transformers.utils import logging
from transformers import MllamaForConditionalGeneration, MllamaVisionModel
from .configuration_legato import LegatoConfig

logger = logging.get_logger(__name__)

class LegatoModel(MllamaForConditionalGeneration):
    """
    This class extends the MllamaForConditionalGeneration model to include a reference to an encoder model.
    """
    config_class = LegatoConfig
    def __init__(
        self,
        config : LegatoConfig,
        load_pretrained_encoder: bool = True
    ):
        super().__init__(config)
        encoder_ref = getattr(config, 'encoder_pretrained_model_name_or_path', None)
        if encoder_ref is not None:
            if load_pretrained_encoder:
                logger.info(f"Loading vision encoder from {encoder_ref}")
                self.vision_model = MllamaVisionModel.from_pretrained(encoder_ref)
                for param in self.vision_model.parameters():
                    param.requires_grad = False
            else:
                self.vision_model = None  # Remove vision model and load it later
        elif load_pretrained_encoder:
            raise ValueError(
                "The configuration does not specify 'encoder_pretrained_model_name_or_path'. "
                "Set load_pretrained_encoder to False to skip loading the encoder."
            )
        else:
            self.vision_model = None
    @classmethod
    @classmethod
    def from_pretrained(cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs
    ):
        # Load the model configuration and weights
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )
        return model
