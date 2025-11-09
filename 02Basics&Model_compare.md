# CS7642 – Lecture 02: Reinforcement Learning Basics

---

## 📌 Introduction

- Reinforcement Learning (RL) is a **conversation** between the agent and the environment.
- The environment reveals itself through **states**.
- The agent:
  - Takes **actions**
  - Receives **rewards**
- The agent **does not know** the environment’s dynamics.
- It learns solely through **continuous interaction**.

---

## 🤖 Behavior Structures

### 🎯 Goal

- Design **learning algorithms** that approximate solutions to decision-making problems.
- Ultimately, we aim to **learn a behavior** that yields **high cumulative rewards**.

### 🧠 Types of Behavior

#### 1. **Plan**

- A fixed sequence of actions.
- Once chosen, the agent executes it without deviation.

✅ Issues:
- The agent doesn’t initially know the right plan.
- It fails in **stochastic** (non-deterministic) environments.

---

#### 2. **Conditional Plan**

- Includes **if-conditions** to change actions based on environmental feedback.
- More adaptable than fixed plans.
- Differs from **dynamic planning**, where the plan is revised if predictions are wrong.

---

#### 3. **Stationary Policy / Universal Plan**

- A **mapping from states to actions**.
- Equivalent to having an "if" clause at **every state**.

✅ Advantages:
- Handles stochasticity well.
- Universally optimal for any **Markov Decision Process (MDP)**.
- Theoretically, there is **always an optimal stationary policy**.

⚠️ Limitation: Can be **very large** in complex environments.

---

## 📊 Evaluating a Policy

### ❓ What Makes a Policy "Optimal"?

- Following a policy generates a **sequence of (state, action, reward)**.
- Due to stochasticity, **many different sequences** can result from the same policy.
- We want to **assign a single value** to a policy to compare it with others.

---

### 🔢 How Do We Compute a Policy's Value?

1. **Convert state transitions to rewards**
   - Evaluate the **immediate reward** at each step.

2. **Truncate the horizon**
   - Sequences are often infinite.
   - In **finite-horizon** tasks, cut off after \( T \) steps.

3. **Summarize a sequence (Return)**
   - Use a **discounted sum of rewards**:
   \[
   G = \sum_{i=1}^{T} \gamma^i r_i
   \]
   Where:
   - \( \gamma \): discount factor (\( 0 < \gamma \leq 1 \))
   - \( r_i \): reward at time \( i \)

4. **Summarize across all sequences**
   - Compute the **expected return** over all possible trajectories:
   \[
   V^\pi(s) = \mathbb{E}_\pi \left[ G \mid s_0 = s \right]
   \]

---

## 📈 Evaluating a Learner

A **good RL learner** produces a **good policy**.

If two learners yield the **same optimal policy**, we differentiate them by:

### 🧮 Computational Complexity
- Time taken to compute the policy.

### 📊 Sample / Experience Complexity
- Number of **interactions** with the environment needed.

> 📦 **Note:** Space complexity is typically **not** a primary concern in RL.

---


# Extras: Model-Free vs Model-Based Algorithms

---

## 🧭 Overview

Reinforcement Learning (RL) agents learn how to act in an environment to maximize cumulative reward.A central distinction in RL is between:

- **Model-Free Methods**: Learn *directly* from trial-and-error interaction.
- **Model-Based Methods**: Learn a *model* of the environment's dynamics and plan actions using it.

---

## 🧠 Model-Free Reinforcement Learning

### 📌 Definition

Model-free RL does **not attempt to model** the transition dynamics or reward function of the environment.
Agents learn **value functions**, **policies**, or both **from direct experience**.

### ✅ Key Characteristics

- Learns by **interacting with the environment**
- Treats the environment as a **black box**
- **Simplicity** and **general applicability**
- Often requires **more real-world data (less sample-efficient)**

### 🔀 Types of Model-Free Learning


| Type           | Description                                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| **On-Policy**  | Learns the value/policy of the**current behavior** being used.                                                 |
| **Off-Policy** | Learns the**optimal policy**, independent of the behavior policy used to collect data. Enables **data reuse**. |

---

### 📘 Common Model-Free Algorithms


| Algorithm                              | Type         | On/Off Policy | Description                                                                                                              |
| ---------------------------------------- | -------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Q-Learning**                         | Value-based  | Off-policy    | Learns Q-values (expected future rewards) for each state-action pair. Can learn the optimal policy even while exploring. |
| **DQN (Deep Q-Network)**               | Value-based  | Off-policy    | Uses deep neural networks to approximate Q-values. Effective in high-dimensional environments like Atari games.          |
| **SARSA**                              | Value-based  | On-policy     | Like Q-learning, but learns the value of the**policy it is currently using**. More conservative updates.                 |
| **REINFORCE**                          | Policy-based | On-policy     | A Monte Carlo policy gradient method. Updates policy based on returns from complete episodes.                            |
| **PPO (Proximal Policy Optimization)** | Policy-based | On-policy     | Uses clipped updates to stabilize training. One of the most commonly used deep RL algorithms.                            |
| **A2C / A3C**                          | Actor-Critic | On-policy     | Combines policy (actor) and value estimation (critic). A3C uses parallel actors to accelerate learning.                  |
| **DDPG**                               | Actor-Critic | Off-policy    | Designed for**continuous action spaces**. Combines DQN and policy gradients.                                             |
| **SAC (Soft Actor-Critic)**            | Actor-Critic | Off-policy    | Incorporates entropy into the objective to promote**exploration and stability** in continuous control tasks.             |

---

## 🧠 Model-Based Reinforcement Learning

### 📌 Definition

Model-based RL **builds an internal model** of the environment’s dynamics and reward function.
It uses this model to **simulate**, **plan**, and **improve learning efficiency**.

### ✅ Key Characteristics

- Learns a **transition model**: \( P(s'|s,a) \)
- Learns a **reward model**: \( R(s,a) \)
- Allows the agent to:
  - **Simulate hypothetical futures**
  - **Plan ahead** before taking actions
- Typically **more sample-efficient**

---

### 🧪 Model-Based Workflow

1. **Learn a model** of environment dynamics.
2. **Simulate outcomes** of possible actions.
3. **Plan** or improve policy using the model (e.g., tree search, rollout).
4. **Act** in the environment using the chosen action.

---

### 📘 Common Model-Based Algorithms


| Algorithm                          | Type                      | Description                                                                                                                                              |
| ------------------------------------ | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dyna-Q**                         | Hybrid                    | Combines Q-learning with a learned model. The agent updates from both real experiences and simulated ones from its model.                                |
| **MPC (Model Predictive Control)** | Planning                  | Plans a short sequence of actions using a learned model, selects the first action, then replans at next step.                                            |
| **AlphaZero / AlphaGo**            | Planning/Hybrid           | Combines neural networks and**Monte Carlo Tree Search (MCTS)**. Learns a model to guide simulations. Dominant in board games.                            |
| **World Models / Dreamer**         | Model Learning / Planning | Learns a**latent representation** of the world, and plans entirely within this “dream” environment. Highly sample-efficient.                           |
| **DreamerV3 (e.g., Minecraft)**    | Model Learning            | Learns to act in complex 3D environments by imagining future trajectories in a**learned latent world**. Used to mine diamonds in Minecraft from scratch. |

---

## 🔄 Comparison Table: Model-Free vs Model-Based RL


| Feature                       | Model-Free RL                             | Model-Based RL                                       |
| ------------------------------- | ------------------------------------------- | ------------------------------------------------------ |
| **Environment Understanding** | No model; learns only from experience     | Builds a model of environment dynamics               |
| **Sample Efficiency**         | Low (needs many real interactions)        | High (can simulate interactions internally)          |
| **Planning Ability**          | No planning; reacts to experience         | Can plan using internal simulations                  |
| **Complexity**                | Conceptually simpler, easier to implement | More complex; requires model learning and validation |
| **Exploration**               | Can be limited without extra techniques   | Model can help explore more effectively              |
| **Popular Examples**          | DQN, PPO, SAC, A3C                        | AlphaZero, Dreamer, MPC, Dyna-Q                      |

---

## 📌 Why Model-Free Is More Common (for Now)

- **Simplicity and generality**: Doesn’t require accurate modeling of dynamics.
- **Strong empirical success** in high-dimensional, complex environments (e.g., video games).
- Easier to implement and scale with deep learning frameworks.

But:🔬 **Model-Based RL is gaining ground** due to:

- Sample efficiency
- Planning capability
- Success in long-horizon and real-world tasks (e.g., robotics, Dreamer in Minecraft)

---

## 🔍 Real-World Analogy


| Approach        | Analogy                                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model-Free**  | A person who learns to ride a bike purely by trial and error — they fall, adjust, and try again.                                                 |
| **Model-Based** | A person who first**learns the physics** of balance and motion, simulates what will happen when they pedal, and practices mentally before trying. |

---

## 🧠 Final Summary


| Category                         | Model-Free                       | Model-Based                                         |
| ---------------------------------- | ---------------------------------- | ----------------------------------------------------- |
| ✅ Learns from experience        | ✔️                             | ✔️                                                |
| 📦 Builds a model of environment | ❌                               | ✔️                                                |
| 🧠 Simulates/plans ahead         | ❌                               | ✔️                                                |
| 🔁 Sample-efficient              | ❌                               | ✔️                                                |
| 🤖 Deep RL examples              | DQN, PPO, SAC                    | AlphaZero, Dreamer, MPC                             |
| 📈 Best for                      | High-dimensional black-box tasks | Tasks where model learning is feasible and valuable |

---

## 📚 Further Reading & Key Papers

- **Sutton & Barto – RL: An Introduction** (Ch. 8–10)
- **“Mastering Chess and Go with AlphaZero” – DeepMind**
- **“World Models” – Ha & Schmidhuber**
- **“DreamerV3” – Hafner et al. (2023)**

---

Let me know if you'd like:

- Flashcards based on these notes
- Diagrams comparing model-free vs model-based pipelines
- A printable `.md` or PDF version of this guide
