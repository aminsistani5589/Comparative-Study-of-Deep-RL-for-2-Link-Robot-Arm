import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)  # logits

def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)

def train(
    num_episodes=1000,
    batch_size=5,
    gamma=0.99,
    lr=1e-3,
    seed=42
):
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    history = []
    batch_losses = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        returns = compute_returns(rewards, gamma)
        # نرمال‌سازی Return در سطح اپیزود
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs)

        # باید شکل هر دو tensor برابر باشد
        assert log_probs.shape == returns.shape

        episode_loss = -(log_probs * returns).mean()
        batch_losses.append(episode_loss)

        episode_return = sum(rewards)
        history.append(episode_return)

        if (ep + 1) % batch_size == 0:
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.clear()

        if (ep + 1) % 50 == 0:
            avg_return = sum(history[-50:]) / 50
            print(f"Episode {ep + 1:4d} | avg return last 50: {avg_return:.1f}")

    env.close()
    return policy, history

def evaluate_render(policy, episodes=3):
    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = policy(obs_t)
                # ارزیابی greedy است، نه sampling
                action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Eval Episode {ep + 1}: return = {total_reward}")
    env.close()

if __name__ == "__main__":
    policy, history = train(
        num_episodes=1000,
        batch_size=5,
        gamma=1.0,
        lr=1e-3,
        seed=42
    )
    # در صورت نیاز برای مشاهده‌ی عملکرد:
    # evaluate_render(policy, episodes=3)
