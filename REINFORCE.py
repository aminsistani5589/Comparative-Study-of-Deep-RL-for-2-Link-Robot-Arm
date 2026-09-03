import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from two_DOF import TwoLinkArm, plot_learning_curve, visualize_policy


# ==================== Policy Network ====================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Mean and log_std layers for continuous Gaussian policy
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        features = self.network(state)
        mean = self.mean_layer(features)
        mean = torch.tanh(mean) * 5.0  # Bound action mean within torque limits [-5, 5]
        std = torch.exp(torch.clamp(self.log_std, min=-20, max=2))
        return mean, std


# ==================== REINFORCE Agent ====================
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def select_action(self, state, deterministic=False):
        """
        Sample an action given the current state.
        
        Args:
            state: Current environment observation.
            deterministic: If True, return mean action directly (for evaluation).
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state)
        
        if deterministic:
            return mean.detach().numpy()[0]
        
        # Sample action from Gaussian distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().numpy()[0], log_prob
        
    def store_transition(self, state, action, reward, log_prob):
        """Store trajectory step for end-of-episode policy update."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def update(self):
        """Compute Monte-Carlo returns and update policy via gradient ascent."""
        if len(self.rewards) == 0:
            return 0.0

        # Compute discounted returns (G_t)
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # Normalize returns for training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        
        return policy_loss.item()


# ==================== Training Loop ====================
def train(env, agent, num_episodes=300):
    episode_rewards = []
    moving_avg_rewards = deque(maxlen=50)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        # Run one full episode
        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.store_transition(state, action, reward, log_prob)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update policy at the end of episode
        loss = agent.update()
        
        episode_rewards.append(episode_reward)
        moving_avg_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(moving_avg_rewards)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(50): {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")
    
    return episode_rewards


# ==================== Main Execution ====================
if __name__ == "__main__":
    # 1. Instantiate environment from two_DOF
    env = TwoLinkArm()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 2. Initialize REINFORCE agent
    agent = REINFORCEAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3, gamma=0.99)
    
    # 3. Train policy
    print("🚀 Starting Training (REINFORCE)...")
    rewards = train(env, agent, num_episodes=300)
    # Save trained policy weights
    torch.save(agent.policy.state_dict(), "reinforce_policy.pth")
    print("Model saved to reinforce_policy.pth")

    # 4. Plot learning curve
    plot_learning_curve(rewards, algorithm_name="REINFORCE", window=10)
    
    # 5. Visualize trained agent behavior
    print("🎬 Visualizing trained policy...")
    test_policy = lambda s: agent.select_action(s, deterministic=True)
    visualize_policy(env, test_policy, num_episodes=3, title="REINFORCE Evaluation")
