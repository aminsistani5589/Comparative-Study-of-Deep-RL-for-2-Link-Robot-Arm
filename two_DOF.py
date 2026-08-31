import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class TwoLinkArm(gym.Env):
    """
    2-DOF Planar Robot Arm Environment
    A 2-link robotic arm tasked with reaching a random target position.
    """
    def __init__(self, link_lengths=[1.0, 1.0]):
        super().__init__()
        
        self.link_lengths = np.array(link_lengths)
        self.dt = 0.05  # Time step
        self.max_torque = 5.0
        
        # Action space: Torque applied to each joint
        self.action_space = spaces.Box(
            low=-self.max_torque, 
            high=self.max_torque, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observation space: [theta1, theta2, theta1_dot, theta2_dot, target_x, target_y, ee_x, ee_y]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,), 
            dtype=np.float32
        )
        
        self.max_steps = 200
        self.current_step = 0
        
        # State variables
        self.joint_angles = np.zeros(2)      # [theta1, theta2]
        self.joint_velocities = np.zeros(2)  # [theta1_dot, theta2_dot]
        self.target_pos = np.zeros(2)        # [target_x, target_y]
        
    def _get_obs(self):
        """Construct the observation vector."""
        ee_pos = self._forward_kinematics()
        obs = np.concatenate([
            self.joint_angles,
            self.joint_velocities,
            self.target_pos,
            ee_pos
        ]).astype(np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial random states."""
        super().reset(seed=seed)
        
        # Random initial joint angles
        self.joint_angles = np.random.uniform(-np.pi, np.pi, 2)
        self.joint_velocities = np.zeros(2)
        
        # Random reachable target position
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.5, 1.8)  # Max reach = sum of link lengths (2.0)
        self.target_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])
        
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def _forward_kinematics(self):
        """Compute end-effector position using forward kinematics."""
        x1 = self.link_lengths[0] * np.cos(self.joint_angles[0])
        y1 = self.link_lengths[0] * np.sin(self.joint_angles[0])
        
        x2 = x1 + self.link_lengths[1] * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y2 = y1 + self.link_lengths[1] * np.sin(self.joint_angles[0] + self.joint_angles[1])
        
        return np.array([x2, y2])
    
    def step(self, action):
        """Execute one step of dynamics and calculate rewards."""
        # Clip action within torque limits
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # Simplified arm dynamics: angular acceleration, friction, and integration
        self.joint_velocities += action * self.dt * 0.5
        self.joint_velocities *= 0.95  # Friction / damping
        self.joint_angles += self.joint_velocities * self.dt
        
        # Normalize angles to [-pi, pi]
        self.joint_angles = (self.joint_angles + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate distance from end-effector to target
        ee_pos = self._forward_kinematics()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        # Reward function: negative distance + bonus for reaching target
        reward = -distance
        if distance < 0.1:
            reward += 10.0
            terminated = True
        else:
            terminated = False
        
        # Control effort penalty
        reward -= 0.01 * np.sum(np.abs(action))
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self, ax=None):
        """Render the 2-DOF arm and target."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Compute joint coordinates
        x1 = self.link_lengths[0] * np.cos(self.joint_angles[0])
        y1 = self.link_lengths[0] * np.sin(self.joint_angles[0])
        
        x2 = x1 + self.link_lengths[1] * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y2 = y1 + self.link_lengths[1] * np.sin(self.joint_angles[0] + self.joint_angles[1])
        
        # Plot links and joints
        ax.cla()
        ax.plot([0, x1], [0, y1], 'b-', linewidth=5, label='Link 1')
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=5, label='Link 2')
        ax.plot(0, 0, 'ko', markersize=10)
        ax.plot(x1, y1, 'go', markersize=10)
        ax.plot(x2, y2, 'ro', markersize=10, label='End-effector')
        ax.plot(self.target_pos[0], self.target_pos[1], 'g*', markersize=20, label='Target')
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('2-DOF Planar Robot')
        
        return ax


def plot_learning_curve(rewards, algorithm_name="RL Algorithm", window=10):
    """Utility function to plot training rewards over episodes."""
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'Moving Avg ({window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{algorithm_name} Training Curve on 2-DOF Robot Arm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_policy(env, policy_fn, num_episodes=3, title="Policy Evaluation"):
    """
    Visualize an agent policy in the environment.
    
    Args:
        env: TwoLinkArm environment instance.
        policy_fn: Function mapping state -> action.
        num_episodes: Number of episodes to display.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done:
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            step_count += 1
            total_reward += reward
            
            # Render current frame
            env.render(ax=ax)
            ax.set_title(f"{title} - Episode {ep + 1} | Step: {step_count} | Reward: {total_reward:.2f}")
            plt.pause(0.01)
        
        plt.pause(0.5)
    
    plt.show()
