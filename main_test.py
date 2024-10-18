from pathlib import Path

import gym_xarm
import gymnasium as gym
import imageio
import numpy
import tinygrad
from tinygrad import Tensor, nn, TinyJit, dtypes
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from huggingface_hub import snapshot_download

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from config import *
from tdmpc_policy import *

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Download the diffusion policy for pusht environment
# pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

# load the dict of safe_tensors
state_dict = safe_load("/Users/msd/Code/experiments/td_mpc/outputs/train/example_xarm_lift_medium/model_5000.safetensors")

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_xarm/XarmLift-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=200,
    #render_mode="rgb_array",
    visualization_width=384,
    visualization_height=384,
)

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.image": [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666],
    "observation.state": [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333],
    "next.reward": [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333],
}
dataset = LeRobotDataset("lerobot/xarm_lift_medium", delta_timestamps=delta_timestamps)
print(dataset.stats)

policy = TDMPCPolicy(TDMPCConfig(), dataset_stats=dataset.stats)
load_state_dict(policy, state_dict)

# Reset the policy and environmens to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)
print(f'numpy_observation: {numpy_observation}')

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())


#@TinyJit
#@Tensor.test()
def test(state:Tensor, image:Tensor) -> Tensor:
    Tensor.no_grad = True
    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    image = image / 255.0
    image = image.permute(2, 0, 1)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    action = policy.select_action(observation)
    print(f'action selected: {action}')
    
    # Prepare the action for the environment
    return action.squeeze(0)

if __name__ == "__main__":
    step = 0
    done = False
    while not done:
        state = Tensor(numpy_observation["agent_pos"], dtype=dtypes.float)
        image = Tensor(numpy_observation["pixels"], dtype=dtypes.float)
        print(f'state: {state}')
        print(f'image: {image}')
        squeezed_action = test(state, image)
    
        # Prepare the action for the environment
        numpy_action = squeezed_action.numpy()
        print(f'numpy_action: {numpy_action}')

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")
