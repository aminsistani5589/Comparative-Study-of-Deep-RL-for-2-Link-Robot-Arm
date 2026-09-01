# 🤖 Comparative Study of Deep Reinforcement Learning for a 2-Link Robot Arm

## 🌟 Overview

Welcome to this repository! This project bridges the gap between **Classical Control Theory** and **Modern Artificial Intelligence** by rigorously evaluating Deep Reinforcement Learning (DRL) algorithms on a classic robotics problem: the 2-Degree-Of-Freedom (2-DOF) planar robot arm.

While traditional control methods (like Computed Torque Control or PID ) are well-established, they often require precise mathematical models of the robot's dynamics and can struggle with unstructured environments or unmodeled disturbances. Deep RL offers a promising, model-free alternative that learns optimal control policies directly from interaction.

However, a critical question remains in the robotics community: **Which DRL algorithm is truly the most efficient, stable, and robust for robotic manipulation tasks?**

## 🚀 Why This Matters?

In the journey toward autonomous robotics, finding algorithms that can safely and efficiently operate within physical torque limits is paramount. This comparative study serves as a foundational step toward more complex robotic tasks, providing clear insights into the trade-offs of different policy-gradient and actor-critic methods in continuous action spaces.

_Whether you are a researcher looking for a clean baseline or a robotics enthusiast curious about Deep RL, this repository provides a transparent, modular, and easy-to-reproduce framework._

## 🤖 Environment: 2-DOF Planar Robot Arm

This project features a custom `gymnasium` environment that simulates a 2-Degree of Freedom (2-DOF) planar robotic arm. The main objective for the Reinforcement Learning agent is to learn to apply the correct joint torques to maneuver the end-effector to a randomly generated target position in a 2D space.

### 📊 Environment Spaces

- **Action Space**: `Box(-5.0, 5.0, (2,), float32)`
  Continuous control representing the torque ($\tau$) applied to each of the two joints.
- **Observation Space**: `Box(-inf, inf, (8,), float32)`
  An 8-dimensional state vector containing:
  - Joint angles: $[\theta_1, \theta_2]$
  - Joint velocities: $[\dot{\theta}_1, \dot{\theta}_2]$
  - Target coordinates: $[x_{target}, y_{target}]$
  - End-effector coordinates: $[x_{ee}, y_{ee}]$

### 🧮 Mathematical Model

The environment uses a simplified physics model to simulate the arm's kinematics and dynamics.

#### Forward Kinematics

Given the link lengths $L_1$ and $L_2$, the positions of the first joint $(x_1, y_1)$ and the end-effector $(x_2, y_2)$ are calculated as follows:

$$x_1 = L_1 \cos(\theta_1)$$
$$y_1 = L_1 \sin(\theta_1)$$
$$x_2 = x_1 + L_2 \cos(\theta_1 + \theta_2)$$
$$y_2 = y_1 + L_2 \sin(\theta_1 + \theta_2)$$

#### Dynamics

The environment operates at a discrete time step of $dt = 0.05s$. We apply a simplified discrete-time integration with a damping/friction coefficient ($\gamma = 0.95$) to simulate natural movement:

$$\dot{\theta}_{t+1} = \left( \dot{\theta}_t + 0.5 \cdot \tau_t \cdot dt \right) \times \gamma$$
$$\theta_{t+1} = \theta_t + \dot{\theta}_{t+1} \cdot dt$$

_(Note: Angles are internally normalized to stay within the [-\pi, \pi] range)._

### 🎯 Reward Function

The reward function is formulated to encourage the agent to reach the target quickly while minimizing control effort (energy consumption). Let $d$ be the Euclidean distance between the end-effector and the target:

$$d = \sqrt{(x_2 - x_{target})^2 + (y_2 - y_{target})^2}$$

The step reward $R_t$ is defined as:

$$R_t = -d - 0.01 \sum_{i=1}^{2} |\tau_i|$$
