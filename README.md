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

## 🧠 Benchmark 2: Deep SARSA (On-Policy Actor-Critic)

Moving from episodic Monte Carlo methods toward Temporal Difference (TD) learning, we implement **Deep SARSA** for continuous control. While traditional tabular SARSA operates on discrete state-action pairs, extending it to our 2-DOF robotic arm requires an **On-Policy Actor-Critic** architecture with continuous action parameterization.

Unlike the greedy, off-policy nature of Q-learning/DQN, Deep SARSA evaluates the return directly from the transitions executed by the current behavioral policy: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

📐 Theory & Formulation
Deep SARSA updates its action-value critic $Q_\phi(s, a)$ by minimizing the single-step bootstrapping TD error:

$$L(\phi) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}, a_{t+1}) \sim \pi_\theta} \left[ \left( Q_\phi(s_t, a_t) - y_t \right)^2 \right]$$

where the on-policy TD target $y_t$ incorporates the _actual next action_ $a_{t+1} \sim \pi_\theta(\cdot | s_{t+1})$ sampled directly from the current policy, rather than a greedy maximization ($\max_{a'} Q$):

$$y_t = r_t + \gamma \, Q_{\phi_{\text{target}}}(s_{t+1}, a_{t+1})$$

The policy network (Actor) $\pi_\theta(a|s)$ models a continuous Gaussian distribution $a \sim \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$ bounded by physical torque limits. It is optimized via the Deterministic/Stochastic Policy Gradient using the estimated Q-values:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s_t \sim \rho^\pi, a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \, Q_\phi(s_t, a_t) \right]$$

### 🏗️ Architectural Choice: Why Actor-Critic?

A fundamental challenge in this implementation is the **continuous action space** $\tau \in [-5.0, 5.0]^2$. In traditional discrete SARSA, the agent selects an action by choosing $a = \arg\max_{a'} Q(s, a')$. However, in a continuous domain, finding this maximum at every single time step is computationally prohibitive as it would require solving an optimization problem within the inner loop of the agent.

To resolve this, we employ an **Actor-Critic architecture**:

1.  **The Actor (Policy Network $\pi_\theta$):** Instead of searching for the best action, we train a dedicated neural network to _parameterize_ the policy. The Actor learns to output the parameters (mean $\mu$ and standard deviation $\sigma$) of a Gaussian distribution, allowing for direct, efficient sampling of continuous torques.
2.  **The Critic (Value Network $Q_\phi$):** The Critic serves as the learned evaluator. It estimates the $Q$-value of the specific state-action pairs $(s, a)$ actually executed by the Actor.

By combining these, we transform the SARSA update into an **On-Policy Actor-Critic** framework. This allows the agent to use the Critic's gradient to "guide" the Actor, effectively shifting the policy towards actions that yield higher predicted $Q$-values, while maintaining the core SARSA requirement of updating based on the _actual_ next action $a_{t+1}$ sampled from the current policy.

<p align="center">
  <img src="https://github.com/aminsistani5589/Comparative-Study-of-Deep-RL-for-2-Link-Robot-Arm/blob/main/DIAGRAMS%20AND%20PICTURES/z1rL_JsXkqZV_9IVPQCB2PF2FmA-OvCCFMLPRN2DmL0dgZgwcQ.png" alt="Actor-Critic Architecture in Deep SARSA" width="550"/>
  <br>
  <em>Figure 2: Overview of the Actor-Critic architecture used in Deep SARSA, illustrating how the Critic evaluates the continuous action generated by the Actor's policy.</em>
</p>

### 🗂️ The Paradox of the Replay Buffer in Deep SARSA

You might wonder: _SARSA is strictly an On-Policy algorithm, so why use an Experience Replay Buffer, a technique usually reserved for Off-Policy methods like DQN?_

The answer lies in solving the fundamental clash between pure RL theory and Neural Network dynamics. We use a **Modified Replay Buffer** for two critical reasons:

1. **Breaking Data Correlation:** In Reinforcement Learning, sequential states and actions are highly correlated. Training a neural network step-by-step on this continuous stream leads to heavily biased gradients and catastrophic instability. By storing experiences in a buffer and sampling random mini-batches, we **break the temporal correlation** between data points, allowing the network to learn robust and stable representations.

2. **Preserving On-Policy Integrity (Storing $a_{t+1}$):** To break correlation without ruining SARSA's on-policy nature, we cannot use a standard buffer. Instead of just saving $(s_t, a_t, r_t, s_{t+1})$, we explicitly store the **next action** ($a_{t+1}$) that the policy _actually committed to_ during the episode. Our stored tuple becomes:
   $$(s_t, a_t, r_t, s_{t+1}, a_{t+1})$$

**The Update Rule:**
When computing the TD Target during batch updates, we do not sample a new action or take a greedy maximum (like Q-learning). Instead, we plug the explicitly stored $a_{t+1}$ directly into the Critic network:

$$y_t = r_t + \gamma \, Q_{\phi_{\text{target}}}(s_{t+1}, a_{t+1})$$

This engineering trick gives us the best of both worlds: the un-correlated stability of batch training, while strictly obeying SARSA's on-policy mathematics!

### 📊 Final Training Logs & Convergence Summary

The training progression stabilized significantly in the final 100 episodes, demonstrating the Critic's steady TD updates and a descending trend in Actor Loss:

| Episode  |    Reward    | Avg Reward (50 Ep) | Critic Loss |  Actor Loss   |
| :------: | :----------: | :----------------: | :---------: | :-----------: |
| **900**  |   $-93.98$   |     $-124.61$      |  $17.6446$  |   $47.8901$   |
| **910**  |  $-107.99$   |     $-130.03$      |  $15.8003$  |   $46.2263$   |
| **920**  |  $-134.57$   |     $-140.23$      |  $17.6443$  |   $46.1282$   |
| **930**  |  $-147.96$   |     $-140.22$      |  $17.1021$  |   $45.5088$   |
| **940**  |   $-92.23$   |     $-130.18$      |  $18.2511$  |   $45.9980$   |
| **950**  |  $-186.52$   |     $-132.66$      |  $18.3096$  |   $42.9766$   |
| **960**  |  $-282.53$   |     $-133.68$      |  $17.8335$  |   $42.9459$   |
| **970**  |  $-127.85$   |     $-122.22$      |  $18.3295$  |   $45.6877$   |
| **980**  |  $-135.24$   |     $-117.49$      |  $16.5598$  |   $42.5635$   |
| **990**  |  $-155.52$   |     $-120.27$      |  $17.4011$  |   $41.0033$   |
| **1000** | **$-78.06$** |   **$-120.07$**    |  $18.2899$  | **$38.5271$** |

🎥 Agent Performance

<p align="center">
  <img src="https://github.com/aminsistani5589/Comparative-Study-of-Deep-RL-for-2-Link-Robot-Arm/blob/main/GIF/sarsa_animation.gif" alt="Deep SARSA 2-DOF Robot Arm" width="600"/>
  <br>
  <em>Figure 3: Trained Deep SARSA policy attempting to stabilize and navigate the 2-DOF robotic arm to target coordinates.</em>
</p>

<div align="center">
  <h2> 🧠 Theoretical Analysis: Deep SARSA vs. REINFORCE </h2>
  <p><i>A Mathematical Deep Dive into Continuous 2-DOF Robotic Control</i></p>
</div>

---

### 🛑 Why Deep SARSA Struggles with 2-DOF Robotic Control

Despite its stable convergence, Deep SARSA is ultimately bottlenecked in highly complex, continuous control tasks like a 2-DOF robotic arm. Mathematically and intuitively, this stems from the **Deadly Triad** and the violation of strict on-policy assumptions caused by the replay buffer.

#### 1. The Staleness of $a_{t+1}$ and Off-Policy Bias

In strict tabular SARSA, the target is calculated using the exact next action $a_{t+1}$ that _will_ be executed by the current policy $\pi_\theta$. However, to stabilize neural networks, we introduced a Replay Buffer storing $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$.

> [!NOTE] > **The Core Issue:** When we sample a mini-batch from this buffer, the stored $a_{t+1}$ was generated by an older version of the policy ($\pi_{\text{old}}$), not the current policy ($\pi_{\text{current}}$).

Mathematically, the intended **on-policy TD target** is:

$$ y*t^{\text{true}} = r_t + \gamma \, Q*{\phi}(s*{t+1}, a*{t+1} \sim \pi*{\text{current}}(\cdot | s*{t+1})) $$

But the **actual calculated target** using the buffer is:

$$ y*t^{\text{buffer}} = r_t + \gamma \, Q*{\phi*{\text{target}}}(s*{t+1}, a*{t+1} \sim \pi*{\text{old}}(\cdot | s\_{t+1})) $$

Because $\pi_{\text{old}} \neq \pi_{\text{current}}$, the update inherently shifts to being **off-policy**.

#### 2. The Deadly Triad Activation

> [!WARNING] > **Triggering the Deadly Triad**
> By combining:
>
> 1. **Function Approximation** (Neural Networks for Actor/Critic)
> 2. **Bootstrapping** (Using $\gamma Q(s_{t+1}, a_{t+1})$ to update $Q(s_t, a_t)$)
> 3. **Off-Policy Updates** (Due to the delayed $a_{t+1}$ in the replay buffer)

We trigger the notorious **"Deadly Triad"** (Sutton & Barto). This leads to unbounded extrapolation errors in Q-value estimations. While we mitigated gradient explosion using a Target Network, the fundamental bias prevents the model from finding the true optimal policy in a continuous state-action manifold, causing it to plateau around a sub-optimal reward of $-120$.

<br>

### 🚀 Why Deep SARSA Still Outperforms REINFORCE

Even with its limitations, Deep SARSA drastically outperformed the vanilla REINFORCE algorithm in both sample efficiency and stability. The proof lies in the mathematical variance of their respective gradient estimators.

#### 1. Variance Reduction through Bootstrapping (TD vs. Monte Carlo)

REINFORCE relies on Monte Carlo (MC) sampling. Its policy gradient update uses the full empirical return $G_t$:

$$ \nabla*\theta J(\theta) = \mathbb{E}*{\pi} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right] $$

_(where $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$)_

> [!IMPORTANT] > **The Variance Problem:** The variance of $G_t$ accumulates over the entire trajectory:
> $$ \text{Var}(G*t) = \text{Var} \left( r_t + \gamma r*{t+1} + \gamma^2 r\_{t+2} + \dots \right) $$
> In a highly sensitive robotic arm environment, a single bad action at step $t=50$ can ruin the trajectory, punishing all good actions that happened before it. This leads to massive variance and destructive gradient updates.

Deep SARSA, however, uses **Temporal Difference (TD) learning**. By utilizing a Critic ($Q_\phi$), it replaces the high-variance $G_t$ with a low-variance bootstrapped estimation:

$$ y*t = r_t + \gamma \, Q*{\phi}(s*{t+1}, a*{t+1}) $$

Since $y_t$ depends only on a single immediate reward $r_t$ and the Critic's expectation of the future, the variance is strictly bounded:

$$ \text{Var}(y_t) \ll \text{Var}(G_t) $$

This allows the Actor to receive consistent, stable feedback (Credit Assignment) at every single step, leading to the smooth, monotonic decrease in Actor Loss observed in our logs.

#### 2. Sample Efficiency and Data Reusability

- **REINFORCE** is strictly episodic; it throws away all $(s, a, r, s')$ data after a single gradient step at the end of the episode.
- **Deep SARSA**, through the use of the `SARSAReplayBuffer`, breaks the temporal correlation of data and allows multiple mini-batch gradient updates per environment step.

This data reusability allows Deep SARSA to converge in a fraction of the environment interactions required by REINFORCE.
