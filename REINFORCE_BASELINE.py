import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from two_DOF import TwoLinkArm, plot_learning_curve, visualize_policy

# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = 8, action_dim: int = 2, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        mean = torch.tanh(self.mean_layer(features)) * 5.0   # clamp to action range
        std = self.log_std.clamp(-20, 2).exp()
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int = 8, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class REINFORCEBaselineAgent:
    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
    ):
        self.gamma = gamma

        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.value_net = ValueNetwork(obs_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.states: list = []
        self.rewards: list = []
        self.log_probs: list = []

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state_t)
        if deterministic:
            return mean.squeeze(0).detach().numpy()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().numpy(), log_prob

    def store_transition(self, state: np.ndarray, reward: float, log_prob: torch.Tensor):
        self.states.append(state)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def _compute_returns(self) -> torch.Tensor:
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self):
        returns = self._compute_returns()
        states_t = torch.FloatTensor(np.array(self.states))

        # Baseline: detach so its gradient doesn't flow into the policy loss
        with torch.no_grad():
            baseline = self.value_net(states_t)

        advantages = returns - baseline
        # Normalize for stable training
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)


        # Policy gradient loss
        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * advantages).sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Value network: MSE against Monte-Carlo returns
        values = self.value_net(states_t)
        value_loss = nn.functional.mse_loss(values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        self.states.clear()
        self.rewards.clear()
        self.log_probs.clear()

        return policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env: TwoLinkArm, agent: REINFORCEBaselineAgent, num_episodes: int = 300):
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, reward, log_prob)
            state = next_state
            total_reward += reward

        policy_loss, value_loss = agent.update()
        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg(10): {avg:8.2f} | "
                f"π Loss: {policy_loss:8.4f} | "
                f"V Loss: {value_loss:8.4f}"
            )

    return episode_rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = TwoLinkArm()

    agent = REINFORCEBaselineAgent(
        obs_dim=8,
        action_dim=2,
        lr=1e-3,
        gamma=0.99,
    )

    print("Training REINFORCE + Baseline on TwoLinkArm...")
    rewards = train(env, agent, num_episodes=300)

    plot_learning_curve(rewards, algorithm_name="REINFORCE + Baseline", window=10)

    visualize_policy(
        env,
        policy_fn=lambda s: agent.select_action(s, deterministic=True),
        num_episodes=3,
        title="REINFORCE + Baseline",
    )

    torch.save(agent.policy.state_dict(), "reinforce_baseline_policy.pth")
    torch.save(agent.value_net.state_dict(), "reinforce_baseline_value.pth")
    print("Weights saved.")