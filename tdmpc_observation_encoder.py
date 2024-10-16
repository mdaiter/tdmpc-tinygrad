from tinygrad import nn, Tensor
import numpy as np
from utils import *
from functools import partial

from config import TDMPCConfig

class TDMPCObservationEncoder():
    """Encode image and/or state vector observations."""

    def __init__(self, config: TDMPCConfig):
        """
        Creates encoders for pixel and/or state modalities.
        TODO: The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.config = config

        if "observation.image" in config.input_shapes:
            self.image_enc_layers: List[Callable[[Tensor], Tensor]] = [
                nn.Conv2d(
                    config.input_shapes["observation.image"][0], config.image_encoder_hidden_dim, 7, stride=2
                ),
                Tensor.relu,
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                Tensor.relu,
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                Tensor.relu,
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                Tensor.relu,
            ]
            dummy_batch = Tensor.zeros(1, *config.input_shapes["observation.image"], requires_grad=False)
            Tensor.no_grad = True
            out_shape = dummy_batch.sequential(self.image_enc_layers).shape[1:]
            Tensor.no_grad = False
            print(f'out_shape: {out_shape}')
            print(f'np.prod(out_shape): {np.prod(out_shape)}')
            print(f'config.latent_dim: {config.latent_dim}')
            print(f'int(np.prod(out_shape)): {int(np.prod(out_shape))}')
            self.image_enc_layers.extend([
                partial(Tensor.flatten, start_dim=1),
                nn.Linear(int(np.prod(out_shape).astype('i')), int(config.latent_dim)),
                nn.LayerNorm(config.latent_dim),
                Tensor.sigmoid,
            ])
        if "observation.state" in config.input_shapes:
            self.state_enc_layers = [
                nn.Linear(config.input_shapes["observation.state"][0], config.state_encoder_hidden_dim),
                Tensor.elu,
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                Tensor.sigmoid,
            ]
        if "observation.environment_state" in config.input_shapes:
            self.env_state_enc_layers = [
                nn.Linear(
                    config.input_shapes["observation.environment_state"][0], config.state_encoder_hidden_dim
                ),
                Tensor.elu,
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                Tensor.sigmoid,
            ]

    def __call__(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        # NOTE: Order of observations matters here.
        if "observation.image" in self.config.input_shapes:
            obs_img_tensor = flatten_forward_unflatten(self.image_enc_layers, obs_dict["observation.image"])
            feat.append(obs_img_tensor)
        if "observation.environment_state" in self.config.input_shapes:
            obs_env_state = obs_dict["observation.environment_state"].sequential(self.env_state_enc_layers)
            feat.append(obs_env_state)
        if "observation.state" in self.config.input_shapes:
            obs_state = obs_dict["observation.state"].sequential(self.state_enc_layers)
            feat.append(obs_state)
        return feat[0].stack(*feat[1:], dim=0).mean(0)
