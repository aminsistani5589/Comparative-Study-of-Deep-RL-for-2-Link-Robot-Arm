# generate_gif.py
"""
Load a saved REINFORCE policy checkpoint and render a GIF of the agent's behavior.
Usage:
    python generate_gif.py                          # uses default paths
    python generate_gif.py --ckpt my_model.pth --out out.gif --eps 5 --fps 25
"""

import argparse
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import imageio
from PIL import Image

from two_DOF import TwoLinkArm


# ── Policy network (must match the one used during training) ──────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        features = self.network(state)
        mean = torch.tanh(self.mean_layer(features)) * 5.0
        std = torch.exp(torch.clamp(self.log_std, min=-20, max=2))
        return mean, std


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, state_dim: int, action_dim: int) -> PolicyNetwork:
    policy = PolicyNetwork(state_dim, action_dim)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return policy


@torch.no_grad()
def get_action(policy: PolicyNetwork, state: np.ndarray) -> np.ndarray:
    """Deterministic greedy action — mean of the Gaussian."""
    state_t = torch.FloatTensor(state).unsqueeze(0)
    mean, _ = policy(state_t)
    return mean.numpy()[0]


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to an HxWx3 uint8 RGB array via PIL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    frame = np.array(img)
    buf.close()
    return frame


# ── GIF generator ─────────────────────────────────────────────────────────────

def generate_gif(
    checkpoint_path: str = "reinforce_policy.pth",
    output_path: str = "reinforce_policy.gif",
    num_episodes: int = 3,
    fps: int = 20,
    pause_frames: int = 10,          # extra frames held at episode end
) -> None:
    env = TwoLinkArm()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = load_policy(checkpoint_path, state_dim, action_dim)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    all_frames: list[np.ndarray] = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            action = get_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Render arm into the axes
            env.render(ax=ax)
            ax.set_title(
                f"REINFORCE  |  Ep {ep + 1}/{num_episodes}"
                f"  |  Step {step}"
                f"  |  Reward {total_reward:.2f}",
                fontsize=10,
            )

            all_frames.append(fig_to_rgb(fig))

        # Hold the last frame briefly between episodes
        if all_frames:
            for _ in range(pause_frames):
                all_frames.append(all_frames[-1].copy())

        print(f"  Episode {ep + 1}: {step} steps, total reward = {total_reward:.2f}")

    plt.close(fig)

    # Write GIF
    imageio.mimsave(output_path, all_frames, fps=fps)
    duration_s = len(all_frames) / fps
    print(f"\nGIF saved → {output_path}  "
          f"({len(all_frames)} frames, {duration_s:.1f} s at {fps} fps)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a GIF from a saved REINFORCE policy.")
    parser.add_argument("--ckpt",   default="reinforce_policy.pth",   help="Path to .pth checkpoint")
    parser.add_argument("--out",    default="reinforce_policy.gif",    help="Output GIF path")
    parser.add_argument("--eps",    type=int, default=3,               help="Number of episodes to render")
    parser.add_argument("--fps",    type=int, default=20,              help="GIF frame rate")
    parser.add_argument("--pause",  type=int, default=10,              help="Extra frames held at episode end")
    args = parser.parse_args()

    generate_gif(
        checkpoint_path=args.ckpt,
        output_path=args.out,
        num_episodes=args.eps,
        fps=args.fps,
        pause_frames=args.pause,
    )
