# sarsa_deep.py
# Deep SARSA for 2-DOF Planar Robot Arm
# On-policy TD control with continuous action space (Gaussian actor + Q-critic)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from two_DOF import TwoLinkArm, plot_learning_curve, visualize_policy



# ==================== Policy Network (Actor) ====================
class PolicyNetwork(nn.Module):
    """Gaussian stochastic policy — identical architecture to REINFORCE baseline."""

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
        mean = torch.tanh(self.mean_layer(features)) * 5.0   # Clip to torque range
        std = torch.exp(torch.clamp(self.log_std, min=-20, max=2))
        return mean, std


# ==================== Q-Network (Critic) ====================
class QNetwork(nn.Module):
    """
    Critic that estimates Q(s, a).
    Input: concatenated [state, action]  →  scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# ==================== SARSA Replay Buffer ====================
class SARSAReplayBuffer:
    """
    Stores (s, a, r, s', a', done) tuples — the key SARSA distinction vs DQN.
    The next action a' is sampled ON-POLICY at interaction time, not re-derived
    greedily at update time.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_action: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(next_actions)),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ==================== Deep SARSA Agent ====================
class DeepSARSAAgent:
    """
    Deep SARSA (on-policy actor-critic variant).

    Critic loss:  MSE( Q(s,a),  r + γ·Q(s',a')·(1-done) )
                  where a' is stored in the replay buffer (on-policy sample).

    Actor loss:   -E[ Q(s, π(s)) ]   (maximize expected Q)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        buffer_capacity: int = 100_000,
    ):
        self.gamma = gamma
        self.batch_size = batch_size

        # Networks
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = QNetwork(state_dim, action_dim)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = SARSAReplayBuffer(capacity=buffer_capacity)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Returns (action_np, log_prob_scalar).
        In deterministic mode returns (mean_np, 0.0) for evaluation.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.actor(state_t)

        if deterministic:
            return mean.detach().cpu().numpy()[0], 0.0

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1).item()
        return action.detach().cpu().numpy()[0], log_prob

    # ------------------------------------------------------------------
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_action: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, next_action, done)

    # ------------------------------------------------------------------
    def update(self):
        """
        One gradient step on both critic and actor.
        Returns (critic_loss_float, actor_loss_float).
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, next_actions, dones = \
            self.replay_buffer.sample(self.batch_size)

        # ---- Critic update (SARSA TD target) ----
        with torch.no_grad():
            # Use the stored next_action (on-policy) — NOT a re-sampled greedy action
            next_q = self.critic(next_states, next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Actor update (maximize Q(s, π(s))) ----
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        sampled_actions = dist.rsample()           # reparameterization for gradients
        actor_loss = -self.critic(states, sampled_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()


# ==================== Training Loop ====================
def train(env: TwoLinkArm, agent: DeepSARSAAgent, num_episodes: int = 300):
    """
    On-policy training loop for Deep SARSA.

    Key difference from off-policy loops: at each step we sample the NEXT
    action from the current policy BEFORE storing the transition, so the
    replay buffer holds true on-policy (s, a, r, s', a', done) tuples.
    """
    episode_rewards = []
    moving_avg_rewards: deque = deque(maxlen=50)

    total_steps = 0
    WARMUP_STEPS = 1000       # Fill buffer before first gradient update

    for episode in range(num_episodes):
        state, _ = env.reset()
        # Sample the FIRST action for this episode
        action, _ = agent.select_action(state)

        episode_reward = 0.0
        critic_loss_ep = 0.0
        actor_loss_ep = 0.0
        update_count = 0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Sample NEXT action on-policy BEFORE storing (the SARSA requirement)
            next_action, _ = agent.select_action(next_state)

            agent.store_transition(state, action, reward, next_state, next_action, done)
            episode_reward += reward
            total_steps += 1

            # Update only after warm-up
            if total_steps >= WARMUP_STEPS:
                c_loss, a_loss = agent.update()
                critic_loss_ep += c_loss
                actor_loss_ep += a_loss
                update_count += 1

            state = next_state
            action = next_action          # SARSA: carry the sampled next action forward

            if done:
                break

        episode_rewards.append(episode_reward)
        moving_avg_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(moving_avg_rewards)
            avg_c = critic_loss_ep / max(update_count, 1)
            avg_a = actor_loss_ep / max(update_count, 1)
            print(
                f"Episode {episode + 1:>4}/{num_episodes} | "
                f"Reward: {episode_reward:>8.2f} | "
                f"Avg(50): {avg_reward:>8.2f} | "
                f"Critic Loss: {avg_c:.4f} | "
                f"Actor Loss: {avg_a:.4f}"
            )

    return episode_rewards

# ==================== Arm Visualization ====================
# Add this import at the top alongside existing imports:
# import matplotlib.pyplot as plt   (add after `import random`)

import matplotlib.pyplot as plt


def visualize_arm(
    env: TwoLinkArm,
    agent: DeepSARSAAgent,
    n_configs: int = 4,
    save_path: str = "sarsa_arm_viz.png",
):
    """
    Runs a deterministic rollout, picks n_configs evenly-spaced frames, and
    renders each as a 2-link arm in a 2×2 dark-theme grid.
    """
    # --- link lengths (fall back gracefully if attribute names differ) ---
    L1 = getattr(env, "l1", getattr(env, "link1_length", 1.0))
    L2 = getattr(env, "l2", getattr(env, "link2_length", 1.0))
    reach = L1 + L2

    def fk(th1: float, th2: float):
        """Forward kinematics → (elbow_x, elbow_y, ee_x, ee_y)."""
        ex  = L1 * np.cos(th1)
        ey  = L1 * np.sin(th1)
        eex = ex + L2 * np.cos(th1 + th2)
        eey = ey + L2 * np.sin(th1 + th2)
        return ex, ey, eex, eey

    # --- collect states from a deterministic rollout ---
    # Typical TwoLinkArm obs layout (8-dim):
    # [cos θ1, sin θ1, cos θ2, sin θ2, θ̇1, θ̇2, goal_x, goal_y]
    frames = []
    state, _ = env.reset()
    for _ in range(500):
        action, _ = agent.select_action(state, deterministic=True)
        next_state, _, terminated, truncated, _ = env.step(action)

        th1 = np.arctan2(float(state[1]), float(state[0]))
        th2 = np.arctan2(float(state[3]), float(state[2]))
        goal = (float(state[6]), float(state[7]))
        ex, ey, eex, eey = fk(th1, th2)
        dist = np.hypot(eex - goal[0], eey - goal[1])
        frames.append((th1, th2, goal, ex, ey, eex, eey, dist))

        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

    # pick n_configs evenly-spaced frames
    stride = max(1, len(frames) // n_configs)
    selected = [frames[i * stride] for i in range(n_configs)]

    # --- plot ---
    BG, PANEL = "#1a1a2e", "#16213e"
    LINK1, LINK2 = "#4e9af1", "#6ab4f5"
    JOINT_BASE, JOINT_ELBOW, JOINT_EE = "#ffffff", "#a0c8ff", "#00d4ff"

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor=BG)
    fig.suptitle(
        "Deep SARSA  —  TwoLink Arm Policy",
        color="#e0e0e0", fontsize=16, fontweight="bold", y=0.98,
    )

    for ax, (th1, th2, goal, ex, ey, eex, eey, dist) in zip(axes.flat, selected):
        ax.set_facecolor(PANEL)
        margin = 0.15
        ax.set_xlim(-reach - margin, reach + margin)
        ax.set_ylim(-reach - margin, reach + margin)
        ax.set_aspect("equal")
        ax.tick_params(colors="#555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        # workspace boundary
        ax.add_patch(plt.Circle((0, 0), reach, color="#20204a", zorder=0))
        ax.add_patch(
            plt.Circle(
                (0, 0), reach, color="#3a3a6a",
                fill=False, linewidth=1.2, linestyle="--", zorder=1,
            )
        )

        # links
        ax.plot([0, ex],  [0, ey],   color=LINK1, linewidth=5,
                solid_capstyle="round", zorder=2)
        ax.plot([ex, eex], [ey, eey], color=LINK2, linewidth=4,
                solid_capstyle="round", zorder=2)

        # joints
        ax.scatter([0],   [0],   color=JOINT_BASE,  s=90, zorder=5)
        ax.scatter([ex],  [ey],  color=JOINT_ELBOW, s=65, zorder=5)
        ax.scatter([eex], [eey], color=JOINT_EE,    s=80, zorder=5)

        # goal
        ax.scatter(*goal, color="#ff4040", marker="*", s=240, zorder=6)

        ax.set_title(
            f"θ₁ = {np.degrees(th1):+.1f}°   θ₂ = {np.degrees(th2):+.1f}°\n"
            f"dist to goal = {dist:.4f}",
            color="#c0d8f0", fontsize=9, pad=5,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"📸 Arm visualization saved → {save_path}")


# ==================== Main Execution ====================
if __name__ == "__main__":
    # 1. Environment
    env = TwoLinkArm()
    state_dim = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.shape[0]       # 2

    # 2. Agent
    agent = DeepSARSAAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        batch_size=256,
    )

    # 3. Train
    print("🚀 Starting Training (Deep SARSA)...")
    rewards = train(env, agent, num_episodes=1000)

    # 4. Save weights
    torch.save(agent.actor.state_dict(), "sarsa_deep_actor.pth")
    torch.save(agent.critic.state_dict(), "sarsa_deep_critic.pth")
    print("✅ Model saved → sarsa_deep_actor.pth  |  sarsa_deep_critic.pth")

    # 5. Learning curve
    plot_learning_curve(rewards, algorithm_name="Deep SARSA", window=10)

    # 6. Visual rollout
    print("🎬 Visualizing trained policy...")
    visualize_policy(env, lambda s: agent.select_action(s, deterministic=True)[0])        # existing two_DOF utility

    # 7. Two-link arm snapshot grid
    visualize_arm(env, agent, save_path="sarsa_arm_viz.png")
