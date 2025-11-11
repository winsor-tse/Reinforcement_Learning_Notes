# 🧩 Partially Observable MDPs (POMDPs) & Applications in Multi-Agent RL

---

## 🎯 Motivation: Why Partial Observability?

In real-world environments, agents typically have **limited and noisy perceptions** of their environment. This makes full state observability infeasible.

**Examples:**
- A robot sees only what's in front of it.
- Poker players can't see opponents' cards.
- Autonomous cars can't observe hidden objects.

---

## 📚 Definition: What is a POMDP?

A **Partially Observable Markov Decision Process (POMDP)** generalizes the MDP by introducing uncertainty in state observations.

A POMDP is defined as a 7-tuple:

$$
(S, A, T, R, \Omega, O, \gamma)
$$

| Symbol | Meaning |
|--------|---------|
| \(S\) | Set of hidden states |
| \(A\) | Set of actions |
| \(T(s'\ s,a)\) | Transition function |
| \(R(s,a)\) | Reward function |
| \(\Omega\) | Set of observations |
| \(O(o s')\) | Observation function |
| \(\gamma\) | Discount factor |

The agent **does not observe \(s\)** directly, but instead gets \(o \sim O(o|s)\).

---

## 📊 Belief States

To act optimally, the agent maintains a **belief** over possible states:
\[
b(s) = P(s | h_t)
\]
Where \(h_t\) is the history of past actions and observations.

### The policy becomes:
\[
\pi: b \rightarrow a
\]

---

## 🧠 Solving POMDPs

| Method | Description |
|--------|-------------|
| **Exact Solutions** | Dynamic programming over belief space (intractable for large problems) |
| **Approximate** | Point-based value iteration, particle filters, policy gradients with memory |
| **Recurrent Policies** | RNNs or LSTMs encode history to simulate belief tracking |

---

## 🧠 Predictive State Representations (PSRs)

An alternative to belief-based POMDPs:
- Instead of modeling **hidden state**, model **future observable events**.
- Represent current state as a vector of **predictions** about future observations given action sequences.

### Why PSRs?

- Operate purely over observable quantities (no hidden latent variable)
- Can be learned from data using supervised learning

---

## 📈 Bayesian Reinforcement Learning (Bayesian RL)

POMDPs and general RL problems often suffer from **uncertainty** in:
- Environment dynamics
- Reward functions

**Bayesian RL** addresses this by treating unknown parameters (e.g., transitions \(T\)) as **random variables with distributions**.

### Benefits:

- Explicitly models **epistemic uncertainty** (lack of knowledge)
- Supports **Bayesian exploration** (e.g., Thompson Sampling)
- Naturally fits **model-based RL** under partial observability

---

### Bayesian POMDPs

Combine belief over environment dynamics with belief over hidden states:
- Posterior over state and model parameters
- Planning involves integrating over both

> Very computationally expensive, but **conceptually elegant**.

---

## 🤖 Deep RL for POMDPs

### Why Deep RL struggles:

- Standard DQNs assume full observability.
- Partial observability violates the Markov assumption.

### Solutions:

| Technique | How it helps |
|----------|--------------|
| **DRQN** (Deep Recurrent Q-Network) | Uses RNN to process observation history |
| **Attention-based RL** | Attends to important past events |
| **Latent Variable Models** | Learn hidden state embeddings (e.g., using VAE) |
| **Variational Inference** | Approximate posterior over hidden states |

---

## 🔄 POMDP in Multi-Agent Settings

In **Multi-Agent Reinforcement Learning (MARL)**:

- Each agent receives **local, private observations**.
- Even if the full state is observable globally, it's **partially observable per agent**.
- This makes MARL essentially a **Decentralized POMDP (Dec-POMDP)** problem.

### Dec-POMDP Definition:

\[
(S, A_1,...,A_n, T, R, \Omega_1,...,\Omega_n, O, \gamma)
\]

- Agents must act based only on **individual observations** \( o_i \sim O_i(s) \)
- Shared or cooperative reward \( R \)

---

## ⚠️ Challenges in Multi-Agent POMDPs

| Challenge | Why it matters |
|----------|----------------|
| **Non-stationarity** | The environment changes as other agents learn |
| **Credit Assignment** | Difficult to attribute success to individual agents |
| **Limited Communication** | Agents can't always share full state info |
| **Exploration** | Coordination under uncertainty is hard |

---

## 🔬 Modern Architectures in Deep Multi-Agent RL

| Algorithm | Description |
|----------|-------------|
| **QMIX** | Factorizes joint Q-value into individual agent utilities |
| **MADDPG** | Centralized critic, decentralized actor (each agent gets unique view) |
| **COMA** | Actor-critic using counterfactual baselines for multi-agent credit |
| **RIAL/DIAL** | RNN-based with learned communication protocols |
| **ROMA** | Role-based policy conditioning to handle partial views and coordination |

---

## 🧪 Use Cases of POMDPs in MARL

| Application | POMDP Feature |
|-------------|----------------|
| **StarCraft micromanagement** | Agents (units) see only local area |
| **Autonomous drones** | Limited sensor input in cluttered environments |
| **Cooperative games (e.g., Hanabi)** | Must infer others' beliefs with minimal info |
| **Warehouse robotics** | Occlusion and sensor noise make state ambiguous |

---

## ✅ Summary Table

| Topic | Key Insight |
|------|-------------|
| POMDP | Models uncertainty in observation |
| Belief State | Probability distribution over true states |
| PSRs | Avoids hidden states by predicting observations |
| Bayesian RL | Treats environment as probabilistic; useful for exploration |
| Dec-POMDP | Core formalism for multi-agent partial observability |
| Deep POMDP Methods | DRQN, Attention, Latent State Models |
| MARL Architectures | QMIX, MADDPG, COMA, DIAL |

---

## 💡 Final Takeaways

> In real-world RL and MARL:
> - Full observability is rare.
> - Memory, inference, and coordination are **non-optional**.
> - POMDPs, PSRs, and Bayesian techniques offer structured ways to tackle **uncertainty**.

---
