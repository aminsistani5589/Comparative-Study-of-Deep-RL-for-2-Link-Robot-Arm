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

---

## 🧠 Benchmark 1: REINFORCE (Monte Carlo Policy Gradient)

To establish our first baseline, we implement the classic **REINFORCE** algorithm (Williams, 1992). As a fundamental Monte Carlo Policy Gradient method, it provides valuable insights into how pure trial-and-error policy optimization behaves on continuous robotic continuous control.

### 📐 Theory & Formulation

Unlike value-based methods, REINFORCE directly optimizes the parameterized policy $\pi_\theta(a|s)$ using stochastic gradient ascent:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

By applying the **Policy Gradient Theorem**, the objective gradient is computed over complete trajectories:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \, G_t \right]$$

where $G_t = \sum_{k=t}^T \gamma^{k-t} R_{k}$ is the discounted return (Monte Carlo return-to-go).

For our continuous action space $\tau \in [-5.0, 5.0]^2$, the policy parameterizes a diagonal Gaussian distribution:
$$a_t \sim \mathcal{N}\left(\mu_\theta(s_t), \Sigma_\theta(s_t)\right)$$

### 🎥 Agent Performance

<p align="center">
  <img src="https://github.com/aminsistani5589/Comparative-Study-of-Deep-RL-for-2-Link-Robot-Arm/blob/main/GIF/reinforce_policy.gif" alt="REINFORCE 2-DOF Robot Arm" width="550"/>
  <br>
  <em>Figure 1: Trained REINFORCE policy controlling the 2-DOF robotic arm to track target coordinates.</em>
</p>

### 📈 Convergence & Observations

During training (300 episodes), we observe typical Monte Carlo behavior:

```text
Episode 270/300 | Reward: -212.87 | Avg(50): -247.56 | Loss: -24.8113
Episode 280/300 | Reward: -250.22 | Avg(50): -234.61 | Loss: 2.1800
Episode 290/300 | Reward: -207.95 | Avg(50): -241.64 | Loss: -5.8923
Episode 300/300 | Reward: -298.70 | Avg(50): -244.07 | Loss: 7.3512
```

### 🚨 The "Disaster" Analysis: Why Did REINFORCE Fail?

As expected for a continuous control robotics task, the vanilla REINFORCE algorithm struggled heavily, leading to unstable learning, erratic joint movements, and ultimately, policy collapse. This "failure" is actually a perfect textbook example of the theoretical limitations of pure Monte Carlo policy gradients. Here is the mathematical and theoretical breakdown of why this disaster happened:

#### 1. The Curse of High Variance (Credit Assignment Problem)

REINFORCE updates the policy based on the full Monte Carlo return:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
The core issue here is the **Credit Assignment Problem**. If the agent applies a brilliant torque at joint 1 at step $t=10$, but makes a terrible move at $t=50$ that ruins the trajectory, the overall return $G_{10}$ becomes severely degraded. The algorithm will unfairly penalize the _good_ action at $t=10$. In a continuous 2-DOF environment with complex dynamics, this high variance in $G_t$ causes the gradient updates to swing wildly in conflicting directions.

#### 2. Adding a Baseline (And Why It Wasn't Enough)

In a purely vanilla formulation, the gradient estimator is:
$$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a|s) \cdot G_t$$
Multiplying the log probability directly by the raw return $G_t$ means the agent doesn't know if an action was actually _better than average_. To counter this, I implemented a Baseline (a Value function $V(s)$) to calculate the **Advantage** ($A_t = G_t - V(s_t)$) instead of using raw returns.

> 💡 **Implementation Note:** I explicitly coded and tested the "REINFORCE with Baseline" variant. While monitoring the training metrics confirmed that the baseline _successfully and mathematically reduced the gradient variance_, **the overall physical results were still just as bad!** The policy still collapsed. This was a crucial empirical finding: merely reducing variance is entirely insufficient for complex continuous control if we do not also address unbounded step-sizes and chaotic exploration.

#### 3. Catastrophic Gaussian Exploration (Uncorrelated Noise)

Our policy outputs actions sampled from a Gaussian distribution $\mathcal{N}(\mu, \sigma)$. In early training, $\sigma$ is naturally large to encourage exploration. However, independently sampling from a Gaussian at every single time step produces extreme, high-frequency jitter. Because this noise is completely uncorrelated over time, the agent fails to explore smooth or coherent trajectories. Instead, this chaotic sequence of disconnected actions throws the agent into unrecoverable or meaningless states, severely hindering its ability to collect useful trajectory data.

#### 4. Step-Size Sensitivity (No Trust Region)

Vanilla Policy Gradient algorithms have no mechanism to restrict the size of the policy update. A single bad batch of trajectories with extreme returns can produce a massive gradient, throwing the neural network weights $\theta$ off a cliff into a parameter space where $\sigma$ collapses to zero or $\mu$ outputs NaNs. Once the policy falls into this "disaster zone," it cannot recover.

🧠 Benchmark 2: Deep SARSA (On-Policy Actor-Critic)
Moving from episodic Monte Carlo methods toward Temporal Difference (TD) learning, we implement **Deep SARSA** for continuous control. While traditional tabular SARSA operates on discrete state-action pairs, extending it to our 2-DOF robotic arm requires an **On-Policy Actor-Critic** architecture with continuous action parameterization.

Unlike the greedy, off-policy nature of Q-learning/DQN, Deep SARSA evaluates the return directly from the transitions executed by the current behavioral policy: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

📐 Theory & Formulation
Deep SARSA updates its action-value critic $Q_\phi(s, a)$ by minimizing the single-step bootstrapping TD error:

$$L(\phi) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}, a_{t+1}) \sim \pi_\theta} \left[ \left( Q_\phi(s_t, a_t) - y_t \right)^2 \right]$$

where the on-policy TD target $y_t$ incorporates the _actual next action_ $a_{t+1} \sim \pi_\theta(\cdot | s_{t+1})$ sampled directly from the current policy, rather than a greedy maximization ($\max_{a'} Q$):

$$y_t = r_t + \gamma \, Q_{\phi_{\text{target}}}(s_{t+1}, a_{t+1})$$

The policy network (Actor) $\pi_\theta(a|s)$ models a continuous Gaussian distribution $a \sim \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$ bounded by physical torque limits. It is optimized via the Deterministic/Stochastic Policy Gradient using the estimated Q-values:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s_t \sim \rho^\pi, a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \, Q_\phi(s_t, a_t) \right]$$

### 🔑 Why Deep SARSA over Monte Carlo?

- **Low Variance via Bootstrapping:** By replacing the high-variance Monte Carlo full return $G_t$ with a 1-step TD estimate ($r_t + \gamma Q$), credit assignment becomes far more localized and stable.
- **Conservative, Safe Exploration:** Because SARSA optimizes with respect to the _current stochastic policy's behavior_ (including exploration noise), it is inherently more conservative around boundary states than off-policy methods—making it an intriguing benchmark for constrained physical systems.

🎥 Agent Performance

<p align="center">
  <img src="assets/deep_sarsa_eval.gif" alt="Deep SARSA 2-DOF Robot Arm" width="600"/>
  <br>
  <em>Figure 2: Trained Deep SARSA policy attempting to stabilize and navigate the 2-DOF robotic arm to target coordinates.</em>
</p>
