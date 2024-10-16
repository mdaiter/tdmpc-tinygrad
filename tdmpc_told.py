from tinygrad import Tensor, nn, dtypes
import numpy as np

from config import TDMPCConfig
from tdmpc_observation_encoder import TDMPCObservationEncoder
from utils import orthogonal_, calculate_gain

class TDMPCTOLD():
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, config: TDMPCConfig):
        super().__init__()
        self.config = config
        self._encoder = TDMPCObservationEncoder(config)
        self._dynamics = [
            nn.Linear(config.latent_dim + config.output_shapes["action"][0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.mish,
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.mish,
            nn.Linear(config.mlp_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            Tensor.sigmoid,
        ]
        self._reward = [
            nn.Linear(config.latent_dim + config.output_shapes["action"][0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.mish,
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.mish,
            nn.Linear(config.mlp_dim, 1),
        ]
        self._pi = [
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.mish,
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.mish,
            nn.Linear(config.mlp_dim, config.output_shapes["action"][0]),
        ]
        self._Qs = []
        for _ in range(config.q_ensemble_size):
            self._Qs.append([
                nn.Linear(config.latent_dim + config.output_shapes["action"][0], config.mlp_dim),
                nn.LayerNorm(config.mlp_dim),
                Tensor.tanh,
                nn.Linear(config.mlp_dim, config.mlp_dim),
                Tensor.elu,
                nn.Linear(config.mlp_dim, 1),
            ])
        self._V = [
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            Tensor.tanh,
            nn.Linear(config.mlp_dim, config.mlp_dim),
            Tensor.elu,
            nn.Linear(config.mlp_dim, 1),
        ]
        self._init_weights(config)

    def _init_weights(self, config: TDMPCConfig):
        """Initialize model weights.

        Orthogonal initialization for all linear and convolutional layers' weights (apart from final layers
        of reward network and Q networks which get zero initialization).
        Zero initialization for all linear and convolutional layers' biases.
        """

        def _apply_fn(m):
            if isinstance(m, nn.Linear):
                m.weight = orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias = Tensor.zeros_like(m.bias)
            elif isinstance(m, nn.Conv2d):
                gain = calculate_gain("relu")
                m.weight = orthogonal_(m.weight, gain)
                if m.bias is not None:
                    m.bias = Tensor.zeros_like(m.bias)

        for m in [*(self._dynamics), *(self._reward), *(self._pi), *(self._V)]:
            _apply_fn(m)
        for list_m in [*(self._Qs)]:
            for m in [*(list_m)]:
                _apply_fn(m)
        if "observation.image" in config.input_shapes:
            for m in [*(self._encoder.image_enc_layers)]: _apply_fn(m)
        if "observation.state" in config.input_shapes:
            for m in [*(self._encoder.state_enc_layers)]: _apply_fn(m)
        if "observation.environment_state" in config.input_shapes:
            for m in [*(self._encoder.env_state_enc_layers)]: _apply_fn(m)

        for list_m in [*(self._Qs)]:
            assert isinstance(
                list_m[-1], nn.Linear
            ), "Sanity check. The last linear layer needs 0 initialization on weights."
            list_m[-1].weight = Tensor.zeros_like(list_m[-1].weight)
            list_m[-1].bias = Tensor.zeros_like(list_m[-1].bias)  # this has already been done, but keep this line here for good measure
        
        for m in [self._reward]:
            assert isinstance(
                m[-1], nn.Linear
            ), "Sanity check. The last linear layer needs 0 initialization on weights."
            m[-1].weight = Tensor.zeros_like(m[-1].weight)
            m[-1].bias = Tensor.zeros_like(m[-1].bias)  # this has already been done, but keep this line here for good measure

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """Encodes an observation into its latent representation."""
        return self._encoder(obs)

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the next state's latent representation and the reward given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            A tuple containing:
                - (*, latent_dim) tensor for the next state's latent representation.
                - (*,) tensor for the estimated reward.
        """
        x = z.cat(a, dim=-1)
        dynamics = x.sequential(self._dynamics)
        reward = x.sequential(self._reward).squeeze(-1)
        return dynamics, reward

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        """Predict the next state's latent representation given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            (*, latent_dim) tensor for the next state's latent representation.
        """
        x = z.cat(a, dim=-1)
        return x.sequential(self._dynamics)

    def pi(self, z: Tensor, std: float = 0.0) -> Tensor:
        """Samples an action from the learned policy.

        The policy can also have added (truncated) Gaussian noise injected for encouraging exploration when
        generating rollouts for online training.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            std: The standard deviation of the injected noise.
        Returns:
            (*, action_dim) tensor for the sampled action.
        """
        action = z.sequential(self._pi).tanh()
        if std > 0:
            std = Tensor.ones_like(action) * std
            action += Tensor.randn(action.shape) * std
        return action

    def V(self, z: Tensor) -> Tensor:  # noqa: N802
        """Predict state value (V).

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
        Returns:
            (*,) tensor of estimated state values.
        """
        return z.sequential(self._V).squeeze(-1)

    def Qs(self, z: Tensor, a: Tensor, return_min: bool = False) -> Tensor:  # noqa: N802
        """Predict state-action value for all of the learned Q functions.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
            return_min: Set to true for implementing the detail in App. C of the FOWM paper: randomly select
                2 of the Qs and return the minimum
        Returns:
            (q_ensemble, *) tensor for the value predictions of each learned Q function in the ensemble OR
            (*,) tensor if return_min=True.
        """
        x = z.cat(a, dim=-1)
        if not return_min:
            return Tensor.stack(*[x.sequential(q).squeeze(-1) for q in self._Qs], dim=0)
        else:
            if len(self._Qs) > 2:  # noqa: SIM108
                Qs = [self._Qs[i] for i in np.random.choice(len(self._Qs), size=2)]
            else:
                Qs = self._Qs
            return Tensor.stack(*[x.sequential(q).squeeze(-1) for q in Qs], dim=0).min(axis=0)[0]
