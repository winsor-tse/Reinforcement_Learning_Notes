# 🧠 CS7642 – Lecture 04: Convergence (with Deep RL Context)

---

## 🎯 Introduction

In reinforcement learning with **control**, we aim to **learn the value of actions**, not just states. This means:

- **Without control**: Agent passively learns from environment transitions.
- **With control**: Agent actively chooses actions and learns optimal behaviors.

Modern Deep RL applies these same ideas using **neural networks to approximate value functions or policies**, but the **theoretical guarantees** still hinge on these foundational convergence concepts.

---

## 📐 Bellman Equations: Foundations of Value Iteration

### 🔹 Value Function (No Actions):

\[
V(s) = R(s) + \gamma \sum_{s'} T(s, s') V(s')
\]

Where:
- \( R(s) \): immediate reward from state \( s \)
- \( T(s, s') \): transition probability to next state
- \( \gamma \): discount factor

---

### 🔹 TD(0) Update (Value-Based Learning):

\[
V_t(s_{t-1}) = V_{t-1}(s_{t-1}) + \alpha_t \left[ r_t + \gamma V_{t-1}(s_t) - V_{t-1}(s_{t-1}) \right]
\]

✅ This update is used in **value-based methods**, such as TD-learning and is foundational to DQN and bootstrapped updates in Deep RL.

---

### 🔹 Q-Learning Bellman Equation (With Actions):

\[
Q(s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s') \max_{a'} Q(s', a')
\]

In **Deep Q-Networks (DQN)**:
- \( Q \) is approximated by a neural network.
- Transitions \( (s, a, r, s') \) are sampled from replay buffer.

---

### 🔹 Q-Learning Update Rule:

\[
Q_t(s_{t-1}, a_{t-1}) = Q_{t-1}(s_{t-1}, a_{t-1}) + \alpha_t \left[ r_t + \gamma \max_{a'} Q_{t-1}(s_t, a') - Q_{t-1}(s_{t-1}, a_{t-1}) \right]
\]

In Deep RL, this becomes:

\[
\theta_t \leftarrow \theta_{t-1} + \alpha \nabla_\theta \left[ r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_{\theta}(s, a) \right]^2
\]

- \( \theta \): parameters of the Q-network
- \( \theta^- \): target network (frozen copy for stability)

---

## 🔁 Bellman Operator & Contraction Mapping

### 🔹 Bellman Operator Definition

Let \( B \) be an operator acting on Q-functions:

\[
[BQ](s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s') \max_{a'} Q(s', a')
\]

- It maps one Q-function to another.
- In Deep RL, this operator guides **value backup** steps.

---

### 🔹 Value Iteration with Bellman Operator

- Fixed point: \( Q^* = BQ^* \)
- Value iteration: \( Q_t = BQ_{t-1} \)

This recursion is used in classic tabular RL and is the **inspiration behind bootstrapped learning** in Deep RL (e.g., DQN, DDPG, SAC).

---

## 📉 Contraction Mapping and Convergence

### ✅ Definition

A Bellman operator \( B \) is a **contraction mapping** if:

\[
\| BF - BG \|_\infty \leq \gamma \| F - G \|_\infty, \quad 0 \leq \gamma < 1
\]

This means:
- Applying \( B \) brings functions **closer together** in sup-norm.
- Guarantees **convergence to a unique fixed point**.

---

### 🔍 Consequences of Contraction:

- \( Q^* = BQ^* \) has a **unique solution**.
- Iteratively applying \( B \) yields convergence:

\[
\| Q_t - Q^* \|_\infty \leq \gamma \| Q_{t-1} - Q^* \|_\infty
\]

➡️ In Deep RL, this concept underpins **why TD learning converges** (in theory).

---

## 🧠 Proof Sketch of Bellman Contraction

Given two Q-functions \( Q_1 \), \( Q_2 \):

\[
\| BQ_1 - BQ_2 \|_\infty = \max_{s,a} \left| BQ_1(s, a) - BQ_2(s, a) \right|
\]

By reward cancellation:

\[
= \gamma \max_{s,a} \left| \sum_{s'} T(s, a, s') \left[ \max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \right] \right|
\]

Since it’s a weighted average:

\[
\leq \gamma \max_{s',a'} | Q_1(s', a') - Q_2(s', a') | = \gamma \| Q_1 - Q_2 \|_\infty
\]

---

## 📌 max Operator is a Non-Expansion

For any functions \( f \), \( g \):

\[
| \max_a f(a) - \max_a g(a) | \leq \max_a | f(a) - g(a) |
\]

This is used in proving that the Bellman operator is a contraction.

---

## ✅ Convergence Theorem (Stochastic Approximation Setting)

Let \( Q^* = BQ^* \) and:

\[
Q_{t+1}(s, a) = (1 - \alpha_t(s, a)) Q_t(s, a) + \alpha_t(s, a) \cdot \text{Target}_t(s, a)
\]

Convergence is guaranteed if:

1. **Only current state-action is updated**:

\[
\alpha_t(s, a) = 0 \quad \text{if} \quad (s, a) \neq (s_t, a_t)
\]

2. **Non-expansion holds** for backup operator:

\[
| [B_t U_1](s,a) - [B_t U_2](s,a) | \leq (1 - \alpha_t(s,a)) | U_1(s,a) - U_2(s,a) |
\]

3. **Contraction property**:

\[
| [B_t U](s,a) - [B_t Q](s,a) | \leq \gamma \alpha_t(s,a) | Q^*(s,a) - Q(s,a) |
\]

4. **Learning rate conditions**:

\[
\sum_t \alpha_t(s,a) = \infty, \quad \sum_t \alpha_t(s,a)^2 < \infty
\]

➡️ These ensure **convergence in expectation** (similar to SGD).

---

## 💡 Modern Deep RL Implications

| Concept | How It Appears in Deep RL |
|--------|----------------------------|
| Bellman Equation | Used in TD targets (e.g., DQN, A2C, SAC) |
| Bellman Operator | Applied in training loops via bootstrapping |
| Contraction Mapping | Justifies convergence (approximate, not guaranteed in deep nets) |
| Value Iteration | Implemented via repeated target updates |
| Learning Rates | Managed by optimizers (e.g., Adam, RMSProp) |
| Update Frequency | In DQN, only minibatch updates are performed, not full backups |

---

### 🔁 Bootstrapping in Deep RL

- **Bootstrapping**: The target includes the estimate of future value (not full return)
  - Example in DQN:
    \[
    y_t = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
    \]
- Similar in **Actor-Critic**:
    \[
    \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
    \]

➡️ These are **approximations** of value iteration using neural networks.

---

## 🧠 Final Thoughts

While Deep RL doesn't strictly meet all theoretical assumptions (e.g., function approximation, non-tabular settings), the **mathematical foundations of convergence** via **contraction mappings** and **Bellman operators** still deeply inform:

- Target construction  
- Network updates  
- Stability analysis  
- Off-policy corrections (e.g., Retrace, Importance Sampling)

Understanding these guarantees is key to designing and debugging reliable Deep RL algorithms.

---
