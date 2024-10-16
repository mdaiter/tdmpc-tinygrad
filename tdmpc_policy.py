import tinygrad
from tinygrad import nn, Tensor
from tinygrad.nn import BatchNorm
from collections import deque
from copy import deepcopy
from functools import partial
from typing import Callable

from config import TDMPCConfig
from normalize import Normalize, Unnormalize
from tdmpc_told import TDMPCTOLD
from utils import flatten_forward_unflatten, random_shifts_aug, linspace, calculate_gain, orthogonal_

def topk(input_:Tensor, k, dim=-1, largest=True, sorted=False):
    k = min(k, input_.shape[dim]-1)
    input_ = input_.numpy()
    if largest: input_ *= -1
    ind = np.argpartition(input_, k, axis=dim)
    if largest: input_ *= -1
    ind = np.take(ind, np.arange(k), axis=dim) # k non-sorted indices
    input_ = np.take_along_axis(input_, ind, axis=dim) # k non-sorted values
    if not sorted: return Tensor(input_), ind
    if largest: input_ *= -1
    ind_part = np.argsort(input_, axis=dim)
    ind = np.take_along_axis(ind, ind_part, axis=dim)
    if largest: input_ *= -1
    val = np.take_along_axis(input_, ind_part, axis=dim)
    return Tensor(val), ind

def update_ema_parameters(ema_net, net, alpha: float):
    Tensor.no_grad = True
    net_state_dict = nn.state.get_state_dict(net)
    ema_net_state_dict = nn.state.get_state_dict(ema_net)
    """Update EMA parameters in place with ema_param <- alpha * ema_param + (1 - alpha) * param."""
    for p_n in net_state_dict:
        if p_n in ema_net_state_dict:
            print(f'Updating EMA param with key: {p_n}')
            p_v = net_state_dict[p_n].detach()
            ema_net_state_dict[p_n] *= alpha
            ema_net_state_dict[p_n] += p_v.cast(dtype=ema_net_state_dict[p_n].dtype) * (1.0 - alpha)
    Tensor.no_grad = False

class TDMPCPolicy():
    """Implementation of TD-MPC learning + inference.

    Please note several warnings for this policy.
        - Evaluation of pretrained weights created with the original FOWM code
            (https://github.com/fyhMer/fowm) works as expected. To be precise: we trained and evaluated a
            model with the FOWM code for the xarm_lift_medium_replay dataset. We ported the weights across
            to LeRobot, and were able to evaluate with the same success metric. BUT, we had to use inter-
            process communication to use the xarm environment from FOWM. This is because our xarm
            environment uses newer dependencies and does not match the environment in FOWM. See
            https://github.com/huggingface/lerobot/pull/103 for implementation details.
        - We have NOT checked that training on LeRobot reproduces the results from FOWM.
        - Nevertheless, we have verified that we can train TD-MPC for PushT. See
          `lerobot/configs/policy/tdmpc_pusht_keypoints.yaml`.
        - Our current xarm datasets were generated using the environment from FOWM. Therefore they do not
          match our xarm environment.
    """

    name = "tdmpc"

    def __init__(
        self, config: TDMPCConfig | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()

        if config is None:
            config = TDMPCConfig()
        self.config = config
        self.model = TDMPCTOLD(config)
        self.model_target = deepcopy(self.model)
        for param in nn.state.get_parameters(self.model_target):
            param.requires_grad = False

        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = lambda x: x
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: This check is covered in the post-init of the config but have a sanity check just in case.
        self._use_image = False
        self._use_env_state = False
        if len(image_keys) > 0:
            assert len(image_keys) == 1
            self._use_image = True
            self.input_image_key = image_keys[0]
        if "observation.environment_state" in config.input_shapes:
            self._use_env_state = True

        self.reset()

    def reset(self):
        """
        Clear observation and action queues. Clear previous means for warm starting of MPPI/CEM. Should be
        called on `env.reset()`
        """
        self._queues = {
            "observation.state": deque(maxlen=1),
            "action": deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self._use_image:
            self._queues["observation.image"] = deque(maxlen=1)
        if self._use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=1)
        # Previous mean obtained from the cross-entropy method (CEM) used during MPC. It is used to warm start
        # CEM for the next step.
        self._prev_mean: tinygrad.Tensor | None = None

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        Tensor.no_grad = True
        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.image"] = batch[self.input_image_key]

        self._queues = populate_queues(self._queues, batch)

        # When the action queue is depleted, populate it again by querying the policy.
        if len(self._queues["action"]) == 0:
            batch = {key: Tensor.stack(*(list(self._queues[key])), dim=1) for key in batch}

            # Remove the time dimensions as it is not handled yet.
            for key in batch:
                assert batch[key].shape[1] == 1
                batch[key] = batch[key][:, 0]

            # NOTE: Order of observations matters here.
            encode_keys = []
            if self._use_image:
                encode_keys.append("observation.image")
            if self._use_env_state:
                encode_keys.append("observation.environment_state")
            encode_keys.append("observation.state")
            z = self.model.encode({k: batch[k] for k in encode_keys})
            if self.config.use_mpc:  # noqa: SIM108
                actions = self.plan(z)  # (horizon, batch, action_dim)
            else:
                # Plan with the policy (π) alone. This always returns one action so unsqueeze to get a
                # sequence dimension like in the MPC branch.
                actions = self.model.pi(z).unsqueeze(0)

            actions = actions.clamp(-1, +1)

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues["action"].append(actions[0])
            else:
                # Action queue is (n_action_steps, batch_size, action_dim), so we transpose the action.
                self._queues["action"].extend(actions[: self.config.n_action_steps])

        action = self._queues["action"].popleft()
        Tensor.no_grad = False
        return action
    
    def plan(self, z: Tensor) -> Tensor:
        """Plan sequence of actions using TD-MPC inference.

        Args:
            z: (batch, latent_dim,) tensor for the initial state.
        Returns:
            (horizon, batch, action_dim,) tensor for the planned trajectory of actions.
        """
        Tensor.no_grad = True

        batch_size = z.shape[0]

        # Sample Nπ trajectories from the policy.
        pi_actions = Tensor.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            batch_size,
            self.config.output_shapes["action"][0]
        )
        if self.config.n_pi_samples > 0:
            _z = z.repeat(self.config.n_pi_samples, 1, 1)
            for t in range(self.config.horizon):
                # Note: Adding a small amount of noise here doesn't hurt during inference and may even be
                # helpful for CEM.
                pi_actions[t] = self.model.pi(_z, self.config.min_std)
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        # In the CEM loop we will need this for a call to estimate_value with the gaussian sampled
        # trajectories.
        z = z.repeat(self.config.n_gaussian_samples + self.config.n_pi_samples, 1, 1)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        mean = Tensor.zeros(
            self.config.horizon, batch_size, self.config.output_shapes["action"][0]
        )
        # Maybe warm start CEM with the mean from the previous step.
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * Tensor.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # Randomly sample action trajectories for the gaussian distribution.
            std_normal_noise = Tensor.randn(
                self.config.horizon,
                self.config.n_gaussian_samples,
                batch_size,
                self.config.output_shapes["action"][0],
            )
            gaussian_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise).clamp(-1, 1)

            # Compute elite actions.
            actions = gaussian_actions.cat(pi_actions, dim=1)
            estimated_value = self.estimate_value(z, actions)
            value = (estimated_value != float("nan")).where(estimated_value, 0)
            elite_idxs = topk(value, self.config.n_elites, dim=0)[1]  # (n_elites, batch), grabbing indices
            elite_value = value.gather(0, elite_idxs)  # (n_elites, batch) - take_along_dim
            # (horizon, n_elites, batch, action_dim)
            elite_actions = actions.gather(1, elite_idxs.reshape(1, *elite_idxs.shape, 1).expand(1, *elite_idxs.shape, 1))

            # Update gaussian PDF parameters to be the (weighted) mean and standard deviation of the elites.
            max_value = elite_value.max(0, keepdim=True)[0]  # (1, batch)
            # The weighting is a softmax over trajectory values. Note that this is not the same as the usage
            # of Ω in eqn 4 of the TD-MPC paper. Instead it is the normalized version of it: s = Ω/ΣΩ. This
            # makes the equations: μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²).
            score = (self.config.elite_weighting_temperature * (elite_value - max_value)).exp()
            score /= score.sum(axis=0, keepdim=True)
            # (horizon, batch, action_dim)
            rearranged_score = score.reshape(*score.shape, 1)
            _mean = (rearranged_score * elite_actions).sum(axis=1)
            _rearranged_mean = _mean.reshape(_mean.shape[0], 1, *_mean.shape[1:]).expand(_mean.shape[0], 1, *_mean.shape[1:])
            _std = rearranged_score * (elite_actions - _rearranged_mean).pow(2).sum(dim=1).sqrt()
            
            # Update mean with an exponential moving average, and std with a direct replacement.
            mean = (
                self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            )
            std = _std.clamp(self.config.min_std, self.config.max_std)

        # Keep track of the mean for warm-starting subsequent steps.
        self._prev_mean = mean

        # Randomly select one of the elite actions from the last iteration of MPPI/CEM using the softmax
        # scores from the last iteration.
        actions = elite_actions[:, score.T.multinomial(1).squeeze(), Tensor.arange(batch_size)]

        Tensor.no_grad = False
        return actions

    def estimate_value(self, z: Tensor, actions: Tensor):
        """Estimates the value of a trajectory as per eqn 4 of the FOWM paper.

        Args:
            z: (batch, latent_dim) tensor of initial latent states.
            actions: (horizon, batch, action_dim) tensor of action trajectories.
        Returns:
            (batch,) tensor of values.
        """
        Tensor.no_grad = True
        # Initialize return and running discount factor.
        G, running_discount = 0, 1
        # Iterate over the actions in the trajectory to simulate the trajectory using the latent dynamics
        # model. Keep track of return.
        for t in range(actions.shape[0]):
            # We will compute the reward in a moment. First compute the uncertainty regularizer from eqn 4
            # of the FOWM paper.
            if self.config.uncertainty_regularizer_coeff > 0:
                regularization = -(
                    self.config.uncertainty_regularizer_coeff * self.model.Qs(z, actions[t]).std(0)
                )
            else:
                regularization = 0
            # Estimate the next state (latent) and reward.
            z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
            # Update the return and running discount.
            G += running_discount * (reward + regularization)
            running_discount *= self.config.discount
        # Add the estimated value of the final state (using the minimum for a conservative estimate).
        # Do so by predicting the next action, then taking a minimum over the ensemble of state-action value
        # estimators.
        # Note: This small amount of added noise seems to help a bit at inference time as observed by success
        # metrics over 50 episodes of xarm_lift_medium_replay.
        next_action = self.model.pi(z, self.config.min_std)  # (batch, action_dim)
        terminal_values = self.model.Qs(z, next_action)  # (ensemble, batch)
        # Randomly choose 2 of the Qs for terminal value estimation (as in App C. of the FOWM paper).
        if self.config.q_ensemble_size > 2:
            
            G += (
                running_discount * terminal_values[Tensor.randint((2,), low=0, high=self.config.q_ensemble_size)].min(axis=0)[0]
            )
        else:
            G += running_discount * terminal_values.min(axis=0)[0]
        # Finally, also regularize the terminal value.
        if self.config.uncertainty_regularizer_coeff > 0:
            G -= running_discount * self.config.uncertainty_regularizer_coeff * terminal_values.std(0)

        Tensor.no_grad = False
        return G

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Run the batch through the model and compute the loss.

        Returns a dictionary with loss as a tensor, and other information as native floats.
        """
        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch = deepcopy(dict(batch))  # shallow copy so that adding a key doesn't modify the original
            batch["observation.image"] = batch[self.input_image_key]
        batch = self.normalize_targets(batch)

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch["action"]  # (t, b, action_dim)
        reward = batch["next.reward"]  # (t, b)
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Apply random image augmentations.
        if self._use_image and self.config.max_random_shift_ratio > 0.0:
            observations["observation.image"] = flatten_forward_unflatten(
                [partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio)],
                observations["observation.image"],
            )

        # Get the current observation for predicting trajectories, and all future observations for use in
        # the latent consistency loss and TD loss.
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon, batch_size = next_observations[
            "observation.image" if self._use_image else "observation.environment_state"
        ].shape[:2]

        # Run latent rollout using the latent dynamics model and policy model.
        # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # gives us a next `z`.
        batch_size = batch["index"].shape[0]
        z_pred = self.model.encode(current_observation)
        z_pred_arr = [z_pred]
        reward_pred_arr = []
        for t in range(horizon):
            z_pred_new, reward_pred_new = self.model.latent_dynamics_and_reward(z_pred_arr[t], action[t])
            z_pred_arr.append(z_pred_new)
            reward_pred_arr.append(reward_pred_new)
        z_preds = z_pred_arr[0].stack(*z_pred_arr[1:], dim=0)
        reward_preds = Tensor.stack(*reward_pred_arr, dim=0)

        # Compute Q and V value predictions based on the latent rollout.
        q_preds_ensemble = self.model.Qs(z_preds[:-1], action)  # (ensemble, horizon, batch)
        v_preds = self.model.V(z_preds[:-1])
        info.update({"Q": q_preds_ensemble.mean().item(), "V": v_preds.mean().item()})

        # Compute various targets with stopgrad.
        Tensor.no_grad = True
        # Latent state consistency targets.
        z_targets = self.model_target.encode(next_observations)
        # State-action value targets (or TD targets) as in eqn 3 of the FOWM. Unlike TD-MPC which uses the
        # learned state-action value function in conjunction with the learned policy: Q(z, π(z)), FOWM
        # uses a learned state value function: V(z). This means the TD targets only depend on in-sample
        # actions (not actions estimated by π).
        # Note: Here we do not use self.model_target, but self.model. This is to follow the original code
        # and the FOWM paper.
        q_targets = reward + self.config.discount * self.model.V(self.model.encode(next_observations))
        # From eqn 3 of FOWM. These appear as Q(z, a). Here we call them v_targets to emphasize that we
        # are using them to compute loss for V.
        v_targets = self.model_target.Qs(z_preds[:-1].detach(), action, return_min=True)
        Tensor.no_grad = False

        # Compute losses.
        # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        temporal_loss_coeffs = Tensor(self.config.temporal_decay_coeff).pow(Tensor.arange(horizon)).unsqueeze(-1)
        # Compute consistency loss as MSE loss between latents predicted from the rollout and latents
        # predicted from the (target model's) observation encoder.
        consistency_loss = (
            (
                temporal_loss_coeffs
                * (z_preds[1:] - z_targets).square().mean(axis=-1)
                # `z_preds` depends on the current observation and the actions.
                * batch["observation.state_is_pad"][0].logical_not().int()
                * batch["action_is_pad"].logical_not().int()
                # `z_targets` depends on the next observation.
                * batch["observation.state_is_pad"][1:].logical_not().int()
            )
            .sum(0)
            .mean()
        )
        # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # rewards.
        reward_loss = (
            (
                temporal_loss_coeffs
                * (reward_preds - reward).square()
                * batch["next.reward_is_pad"].logical_not().int()
                # `reward_preds` depends on the current observation and the actions.
                * batch["observation.state_is_pad"][0].logical_not().int()
                * batch["action_is_pad"].logical_not().int()
            )
            .sum(0)
            .mean()
        )
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        q_value_loss = (
            (
                temporal_loss_coeffs
                * (q_preds_ensemble - q_targets.repeat(q_preds_ensemble.shape[0], 1, 1)).square().sum(0)  # sum over ensemble
                # `q_preds_ensemble` depends on the first observation and the actions.
                * batch["observation.state_is_pad"][0].logical_not().int()
                * batch["action_is_pad"].logical_not().int()
                # q_targets depends on the reward and the next observations.
                * batch["next.reward_is_pad"].logical_not().int()
                * batch["observation.state_is_pad"][1:].logical_not().int()
            )
            .sum(0)
            .mean()
        )
        # Compute state value loss as in eqn 3 of FOWM.
        diff = v_targets - v_preds
        # Expectile loss penalizes:
        #   - `v_preds <  v_targets` with weighting `expectile_weight`
        #   - `v_preds >= v_targets` with weighting `1 - expectile_weight`
        raw_v_value_loss = (diff.detach() > 0.0).where(
            self.config.expectile_weight, (1.0 - self.config.expectile_weight)
        ).float() * diff.pow(2).float()
        v_value_loss = (
            (
                temporal_loss_coeffs
                * raw_v_value_loss
                # `v_targets` depends on the first observation and the actions, as does `v_preds`.
                * batch["observation.state_is_pad"][0].logical_not().int()
                * batch["action_is_pad"].logical_not().int()
            )
            .sum(0)
            .mean()
        )

        # Calculate the advantage weighted regression loss for π as detailed in FOWM 3.1.
        # We won't need these gradients again so detach.
        z_preds = z_preds.detach()
        # Use stopgrad for the advantage calculation.
        Tensor.no_grad = True
        advantage = self.model_target.Qs(z_preds[:-1], action, return_min=True) - self.model.V(
            z_preds[:-1]
        )
        info["advantage"] = advantage[0]
        # (t, b)
        exp_advantage = (advantage * self.config.advantage_scaling).exp().clamp(max_=100.0)
        Tensor.no_grad = False
        action_preds = self.model.pi(z_preds[:-1])  # (t, b, a)
        # Calculate the MSE between the actions and the action predictions.
        # Note: FOWM's original code calculates the log probability (wrt to a unit standard deviation
        # gaussian) and sums over the action dimension. Computing the (negative) log probability amounts to
        # multiplying the MSE by 0.5 and adding a constant offset (the log(2*pi)/2 term, times the action
        # dimension). Here we drop the constant offset as it doesn't change the optimization step, and we drop
        # the 0.5 as we instead make a configuration parameter for it (see below where we compute the total
        # loss).
        mse = (action_preds - action).square().sum(-1)  # (t, b)
        # NOTE: The original implementation does not take the sum over the temporal dimension like with the
        # other losses.
        # TODO(alexander-soare): Take the sum over the temporal dimension and check that training still works
        # as well as expected.
        pi_loss = (
            exp_advantage
            * mse
            * temporal_loss_coeffs
            # `action_preds` depends on the first observation and the actions.
            * batch["observation.state_is_pad"][0].logical_not().int()
            * batch["action_is_pad"].logical_not().int()
        ).mean()

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * q_value_loss
            + self.config.value_coeff * v_value_loss
            + self.config.pi_coeff * pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss.item(),
                "reward_loss": reward_loss.item(),
                "Q_value_loss": q_value_loss.item(),
                "V_value_loss": v_value_loss.item(),
                "pi_loss": pi_loss.item(),
                "loss": loss,
                "sum_loss": loss.item() * self.config.horizon,
            }
        )

        # Undo (b, t) -> (t, b).
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return info

    def update(self):
        """Update the target model's parameters with an EMA step."""
        # Note a minor variation with respect to the original FOWM code. Here they do this based on an EMA
        # update frequency parameter which is set to 2 (every 2 steps an update is done). To simplify the code
        # we update every step and adjust the decay parameter `alpha` accordingly (0.99 -> 0.995)
        update_ema_parameters(self.model_target, self.model, self.config.target_model_momentum)

