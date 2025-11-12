# 🎯 Derivation of Policy Gradient Methods from REINFORCE

---

## 🧩 Motivation

Unlike value-based algorithms such as **Q-Learning** or **SARSA**, which estimate value functions and derive policies indirectly,  
**Policy Gradient (PG)** methods directly **optimize the parameters of a stochastic policy** to maximize expected reward.

Policy gradient methods directly optimize a parameterized policy, often represented by a neural network, to maximize expected returns. REINFORCE is a foundational policy gradient algorithm


This approach is particularly useful when:
- The action space is continuous.
- The policy must be stochastic.
- We want smooth updates that generalize better.

---


## DQN vs PPO

PPO (Proximal Policy Optimization) is a policy gradient method that uses a stochastic policy, but DQN (Deep Q-Network) is a value-based method that learns a value function and does not use a policy gradient. DQN vs. PPO Policies DQN (Value-Based): DQN learns the optimal action-value function, \(Q(s,a)\), which estimates the maximum expected future reward for taking an action \(a\) in state \(s\). The policy is implicitly derived by always choosing the action with the highest Q-value, which is a deterministic policy at evaluation time (though an \(\epsilon \)-greedy policy is used for exploration during training).PPO (Policy Gradient): PPO directly learns a parameterized policy function that outputs a probability distribution over actions. This is a stochastic policy, meaning it provides the probabilities of taking each possible action from a given state. The Role of Stochastic Policies Policy gradient methods like PPO use stochastic policies because they directly optimize the probabilities of actions. Policies give the probabilities of taking actions in any state. This approach offers several key advantages: Exploration: The inherent randomness in a stochastic policy encourages continuous exploration of the environment, which can help in avoiding local optima and discovering better long-term strategies.
Handling Continuous Action Spaces: Policy gradient methods are effective in environments with high-dimensional or continuous action spaces (e.g., controlling a robot arm), where enumerating all possible actions (as required by value-based methods like DQN) is impossible.Smoother Policy Updates: Using a probability distribution allows for smoother policy updates, making the learning process more stable compared to the hard (and potentially unstable) shifts in policy that can occur in value-based methods when Q-values chang

## 🧮 Objective Function

Let a policy be parameterized by parameters: 
$$ 
\theta 
$$  
$$ 
\pi_\theta(a|s)
$$
We define a trajectory 
$$tau = (s_0, a_0, s_1, a_1, ..., s_T)$$
and the total return of that trajectory as:
$$ R(\tau) = \sum_{t=0}^{T-1} r(s_t, a_t) $$
Our optimization goal is:
$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [ R(\tau) ]$$


---

## ⚙️ Step 1: Express Gradient of Expected Return

$$
\nabla_\theta J = \nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} [ R(\tau) ]
= \sum_{\tau} \nabla_\theta p_\theta(\tau) R(\tau)
$$

But since we can’t compute over **all possible trajectories**, we’ll rewrite this as an expectation over trajectories that can be **sampled**.

---

## ⚙️ Step 2: Likelihood Ratio Trick (Expanded Derivation)

We want to compute the gradient of the expected return with respect to the policy parameters \( \theta \):


$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} [ R(\tau) ]$$

Because this expectation is over a **distribution that depends on \( \theta \)**, we must take the gradient of an expectation **over a parameterized distribution**.

We apply the **log-derivative trick**, also known as the **likelihood ratio trick**, using the identity:


$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$


> 🔍 **Why this works:**  
> This is the chain rule applied in reverse:

$$\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
> \quad \Rightarrow \quad
> \nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$


---

### 🔄 Apply to Objective


$$\nabla_\theta J(\theta)
= \nabla_\theta \sum_{\tau} p_\theta(\tau) R(\tau)
= \sum_{\tau} \nabla_\theta p_\theta(\tau) R(\tau)$$


Now insert the identity:


$$= \sum_{\tau} p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau)$$


Now express this as an expectation:


$$= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]$$


---


$$\nabla_\theta J(\theta) =
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]$$


This is the core equation behind **REINFORCE** and all **Policy Gradient methods**.

---

### 🔁 Why This is Useful:

This allows us to:
- **Sample trajectories** 
  $$ \tau \sim p_\theta(\tau) $$
- Compute an **unbiased Monte Carlo estimate** of the policy gradient
- Update theta using only the **log-likelihood of actions taken**, without requiring gradients of the environment dynamics

This avoids the need to compute gradients of 
$$p(s_{t+1} | s_t, a_t)$$
which are typically unknown in model-free RL.

---


## ⚙️ Step 3: Expanding the Trajectory Probability

In Step 2, we derived:


$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]$$


To move forward, we expand 
$$\log p_\theta(\tau)$$
based on the **trajectory distribution**.

A trajectory 
$$\tau = (s_0, a_0, s_1, a_1, ..., s_T)$$
has the probability under the current policy:

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t)$$

Taking the logarithm:

$$\log p_\theta(\tau) = \log p(s_0)
+ \sum_{t=0}^{T-1} \left[ \log \pi_\theta(a_t | s_t)
+ \log p(s_{t+1} | s_t, a_t) \right]$$


Since the environment dynamics 
$$p(s_{t+1} | s_t, a_t) p(s_0)$$
are **independent of \( \theta \)**, their gradients are zero:


$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t)$$


Now substitute back into the gradient of the objective:


$$\nabla_\theta J(\theta) =
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[
\left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \right)
R(\tau)
\right]$$


This gives us the REINFORCE form:

---

### ✅ REINFORCE Gradient Estimate


$$\nabla_\theta J(\theta) =
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[
\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)
\right]$$


This version uses the **total return \( R(\tau) \)** for all time steps — which introduces **high variance**, especially in long-horizon tasks.

---

## ⚙️ Step 4: Monte Carlo Estimation Used for REINFORCE and Reward-to-Go

To actually perform learning, we **sample trajectories** using pi_theta and estimate the expectation via Monte Carlo.
Can't compute the expectation exactly. The number of possible trajectories 𝜏 is enormous or infinite. The transition dynamics may be unknown. So we can’t enumerate or sum over all possible. Monte Carlo methods let us estimate an expectation by sampling from the distribution. So instead of computing over all trajectories, we just run the policy 
N times, collect the results, and take the average. Let N  be the number of trajectories. Then:

By the law of large number, we know that the estimated gradient in eq. (0.0.90) is an unbiased estimate of the true policy gradient.
Therefore, we can run stochastic gradient ascent with this estimated
gradient. This forms the basis of the REINFORCE (Algorithm 18)

Start with an arbitrary initial policy pq
while not converged do
    Run simulator with pq to collect trajectories from i=1 to N
    Compute estimated gradient
$$
\nabla_\theta J(\theta)
\approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})
$$

Update parameters Theta <- Theta + gamma DElta J
end
return pi_theta

The REINFORCE algorithm. In step 1, we run the simulator using the current policy to collect
training sequences. In step 2, we approximate the expectation by the
sample mean. Step 3 is the update rule of the algorithm with a being
the step size. The algorithm is then repeated until convergence or
until you are bored.


### ⚠️ Problem: High Variance

Each action gradient is scaled by the **same full return** \( R(\tau) \), regardless of when it occurred. But in reality:

> An action at time \( t \) only affects **future** rewards, not past ones.

---

### 🎯 Solution: Use Reward-to-Go

Define the **reward-to-go** from time step \( t \):


$$R_t = \sum_{t'=t}^{T-1} r(s_{t'}, a_{t'})$$


Substitute this into the policy gradient:


$$\nabla_\theta J(\theta) =
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[
\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) R_t
\right]$$


This version has **lower variance** because it avoids attributing early actions to rewards that occurred long after their effects ended.

---

### ✅ REINFORCE with Reward-to-Go (Final Version)


$$\nabla_\theta J(\theta) =
\mathbb{E} \left[
\sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t | s_t)
\left( \sum_{t'=t}^{T-1} r(s_{t'}, a_{t'}) \right)
\right]$$


This is the most common formulation of **REINFORCE with reward-to-go**, used in many practical implementations and textbooks.

---

## 🧠 Why This Matters in Practice

| Technique            | Purpose                     |
|---------------------|-----------------------------|
| Full return R(\tau) | High variance, but unbiased |
| Reward-to-Go R_t    | Reduced variance, better learning signal |
| Baselines (next step)     | Further reduce variance |

In the next derivation step (Step 5), we’ll introduce a **baseline** 
$$V^\pi(s_t)$$
to further reduce the variance of the estimator, leading to **Advantage Actor–Critic** style updates.

---


## ⚠️ Problem: High Variance

The estimator is **unbiased**, but can have **very high variance**, especially for long episodes or sparse rewards.

---

## ⚙️ Step 5: Reduce Variance with “Reward-to-Go”

Actions at time 
$$t$$
only affect **future rewards**, not past ones.  
Thus, we replace
$$R(\tau)$$
with the **reward-to-go**:

$$R_t = \sum_{t'=t}^{T-1} r(s_{t'}, a_{t'})$$

$$\nabla_\theta J = \mathbb{E}_{\tau} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]$$


---

## ⚙️ Step 6: Baseline Subtraction (Variance Reduction)

We can subtract a **baseline** 
$$b(s_t)$$
without biasing the gradient, since:


$$\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t)] = 0$$


This gives:

$$\nabla_\theta J = \mathbb{E}_\tau
\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) (R_t - b(s_t)) \right]$$


Common choice: 
$$b(s_t) = V^{\pi}(s_t)$$

---

## 🧮 Policy Gradient Theorem

We can replace 
$$R_t - b(s_t)$$
with the **advantage function**:
$$A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$$


Then:

$$ \nabla_\theta J = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\theta}
[ \nabla_\theta \log \pi_\theta(a|s) A^{\pi}(s, a) ] $$

This is the **Policy Gradient Theorem**, the foundation for all modern policy-based methods.

---

## 🧠 Step 7: From REINFORCE → Actor–Critic

The baseline 
$$ b(s_t) $$
can be **learned** by a separate neural network (the “critic”) estimating (either V or Q)
$$V^{\pi}(s_t)$$
$$Q^{\pi}(s_t,a_t)$$

- **Actor:** updates the policy (uses policy gradient)
- **Critic:** estimates value function (bootstraps returns)

This leads to **Actor–Critic algorithms**, which reduce variance and stabilize training.

---

## ⚙️ Step 8: Natural Policy Gradient (NPG)

- Standard gradient ascent depends on parameterization.
- **Natural gradient** rescales updates by the **Fisher Information Matrix (FIM):**


$$ G(\theta) = \mathbb{E}_{s,a \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(a|s)
\nabla_\theta \log \pi_\theta(a|s)^T ]$$


Update rule:

$$\Delta \theta = \eta G^{-1}(\theta) \nabla_\theta J$$


This ensures updates correspond to small **changes in policy**, not parameters.


# 🚀 From Policy Gradient Theorem to Natural Policy Gradient

---

## 🧩 The Policy Gradient Theorem

The **Policy Gradient Theorem** provides a foundational way to express how to change a policy’s parameters to increase expected returns.

### 🎯 Theorem Statement:


$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}
\left[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \right]$$


Where:
- $$d^\pi(s)$$
-  discounted state visitation distribution  
- $$Q^\pi(s,a)$$
-  expected return for taking a in s 
- $$\pi_\theta(a|s)$$
-  parameterized stochastic policy  

✅ This removes any dependence on environment transitions \( $$p(s'|s,a)$$ \), making it a **model-free** result.

---

## 💡 From Q to Advantage

We can reduce variance by subtracting a **baseline** that doesn’t depend on the action:
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$


Then the gradient becomes:
$$\nabla_\theta J(\theta) = \mathbb{E}_{s,a}
\left[ \nabla_\theta \log \pi_\theta(a|s) A^\pi(s,a) \right]$$


✅ This is the practical form used in **Actor–Critic** and **Advantage-based** algorithms.

---

## 🧠 Actor–Critic Framework

Because
$$ Q^\pi(s,a)$$ 
or 
$$ A^\pi(s,a)$$
are typically unknown, we learn them with a separate **critic network**:

- **Actor:** updates the policy parameters theta
- **Critic:** estimates 
- $$V^\pi(s)$$
- $$Q^\pi(s,a)$$

The critic’s value predictions reduce the variance of policy gradient estimates while maintaining unbiased updates.

---

## 🔁 Example: Discrete Softmax Policy

For a differentiable stochastic policy (e.g., softmax/Boltzmann):
$$\pi_\theta(a|s) = \frac{\exp(h_\theta(s,a))}{\sum_{a'} \exp(h_\theta(s,a'))}$$

Then:
$$\nabla_\theta \log \pi_\theta(a|s)
= \nabla_\theta h_\theta(s,a)
- \sum_{a'} \pi_\theta(a'|s) \nabla_\theta h_\theta(s,a')$$

This tells us how a small change in parameters \( \theta \) changes the probability of each action.

---

## ⚙️ Policy Improvement Example Using Boltzmann Policy

Let’s illustrate how **policy gradient algorithms improve policies** in practice through a concrete **actor–critic example**.

---

### 🔹 Setup

We have one state \( s \) and two possible actions \( a_0 \) (bad) and \( a_1 \) (good).  
There is a **feature vector** \( f(s,a) \) and a **parameter** \( q \) controlling the policy.

We use a **Boltzmann (softmax) policy**:
$$p_q(a|s) = \frac{\exp[q^\top f(s,a)]}{\sum_{a'} \exp[q^\top f(s,a')]}$$

Suppose:
$$f(s,a_0) = 3, \quad f(s,a_1) = 1, \quad q = 1$$

Then the action probabilities are:
$$p_q(a_0|s) = \frac{e^{3}}{e^3 + e^1}
= \frac{e^2}{e^2 + 1} \approx 0.88$$

$$p_q(a_1|s) = \frac{e^{1}}{e^3 + e^1}
= \frac{1}{e^2 + 1} \approx 0.12$$

---

### 🔹 Critic Estimates (Value Function)

Let the critic provide the following value estimates:
$$Q^\pi(s,a_0) = 1, \quad Q^\pi(s,a_1) = 100$$

So the “good” action has much higher expected reward.

---

### 🔹 Gradient of the Log Probability

The policy gradient requires:
$$\nabla_q \log p_q(a|s)
= f(s,a) - \mathbb{E}_{a' \sim p_q}[f(s,a')]$$

The expected feature value:
$$\mathbb{E}_{a' \sim p_q}[f(s,a')]
= 0.88 \times 3 + 0.12 \times 1 = 2.76$$

---

### 🔹 Computing the Policy Gradient

The gradient estimate is:
$$\nabla_q J = \mathbb{E}_{a \sim p_q(a|s)}
[\nabla_q \log p_q(a|s) Q^\pi(s,a)]$$

Expanding:
$$\nabla_q J \approx
0.88 \times (3 - 2.76) \times 1
+ 0.12 \times (1 - 2.76) \times 100$$

$$\nabla_q J \approx 0.21 - 21.00 = -20.79$$

Thus:
$$\nabla_q J \approx -20.8$$

---

### 🔹 Interpretation

Since the gradient is **negative**, the update:
$$q \leftarrow q + \alpha \nabla_q J$$

will **decrease** \( q \).  
Lowering \( q \) reduces the relative probability of the “bad” high-feature action \( a_0 \) and increases probability of \( a_1 \), the high-reward action.

✅ The policy has therefore **improved** — it now favors actions that yield higher returns.

---

## 🧩 Summary of Boltzmann Policy Gradient Behavior

| Action | Feature f(s,a)  | Q-value Q(s,a) | Probability | Effect of Update |
|--------|----------------------|----------------------|--------------|------------------|
|  a_0 | 3 | 1 | 0.88 | Probability decreases |
| a_1 | 1 | 100 | 0.12 | Probability increases |

This simple example shows **how gradient ascent on expected return directly reshapes the policy** — increasing probabilities of high-reward actions and decreasing poor ones.

---

## 📉 Highly Correlated Features and Gradient Instability

If the features f(s,a) are highly correlated:
- The Fisher Information Matrix becomes ill-conditioned.
- Gradients oscillate between directions.
- Learning becomes unstable.

Hence, we need a way to **normalize the geometry of updates**.

---

## 🧠 Natural Policy Gradient (NPG)

The **Natural Policy Gradient** corrects this by scaling the gradient by the **inverse of the Fisher Information Matrix (FIM):**

$$\tilde{\nabla}_\theta J = F^{-1}(\theta) \nabla_\theta J$$

Where:

$$F(\theta) = \mathbb{E}_{s,a} \left[
\nabla_\theta \log \pi_\theta(a|s)
\nabla_\theta \log \pi_\theta(a|s)^T
\right]$$

This ensures updates correspond to small changes in **policy behavior**, not just raw parameters.

---

## 🎯 Update Rule
$$\theta_{k+1} = \theta_k + \alpha F^{-1}(\theta_k) \nabla_\theta J$$

- Moves in the direction of steepest ascent in **policy space**
- More stable and **invariant to reparameterization**

---

## 🔗 Connection to KL Regularization and TRPO

- **TRPO (Trust Region Policy Optimization)** enforces:
  $$D_{KL}(\pi_{\theta_{old}} || \pi_\theta) \le \delta$$
- Natural gradients implicitly respect this constraint.
- **ACTOR** approximates the FIM efficiently for large neural networks.

---

## ✅ Summary

| Concept | Description |
|----------|--------------|
| **Policy Gradient Theorem** | Core formula linking policy improvement to expected returns |
| **Advantage Function** | Variance reduction baseline |
| **Actor–Critic** | Learns both policy and value |
| **Boltzmann Policy Example** | Demonstrates gradient-driven improvement |
| **Highly Correlated Features** | Cause instability in vanilla PG |
| **Natural Policy Gradient** | Uses Fisher metric to scale updates |
| **TRPO / ACKTR** | Approximate natural gradient in deep RL |

---

> 🧠 **Final Insight:**  
> Policy gradient methods **directly optimize behavior**, not just values.  
> From simple Boltzmann updates to curvature-aware NPG, they form the mathematical backbone of **modern deep reinforcement learning** methods such as PPO, TRPO, and SAC.


## 🧩 Step 9: Connection to Modern Deep RL

| Algorithm | Relation to Policy Gradient |
|------------|-----------------------------|
| **REINFORCE** | Monte Carlo estimator of ∇J |
| **Actor–Critic (A2C, A3C)** | Adds learned baseline (critic) |
| **PPO (Proximal Policy Optimization)** | Uses clipped surrogate loss for stable updates |
| **TRPO (Trust Region Policy Optimization)** | Constrains step size using KL divergence (approx. NPG) |
| **DDPG / SAC** | Deterministic or entropy-regularized variants for continuous control |

##  Modern Deep RL

- **A2C / A3C (Asynchronous Advantage Actor–Critic)**
- **PPO (Proximal Policy Optimization)** – stable, first-order approximation of TRPO
- **TRPO (Trust Region Policy Optimization)** – constrained optimization using KL-divergence
- **SAC (Soft Actor–Critic)** – entropy-regularized policy gradients for exploration
- **DDPG (Deep Deterministic Policy Gradient)** – deterministic continuous variant of PG
- **TD3** – addresses function approximation instability in DDPG

---

---

## 🧠 Intuition Summary

| Concept | Purpose |
|----------|----------|
| Likelihood Ratio Trick | Express gradient as expectation |
| Reward-to-Go | Reduce variance |
| Baseline / Value Function | Further variance reduction |
| Advantage Function | Emphasizes good vs average actions |
| Actor–Critic | Learn policy and value simultaneously |
| Natural Gradient | Update in policy space, not parameter space |
| PPO/TRPO | Practical improvements for stable learning |

---

## ✅ Key Equations Summary

### REINFORCE gradient
$$\nabla_\theta J = \mathbb{E}[ \nabla_\theta \log \pi_\theta(a|s) R ] $$

### Baseline / variance reduction 
$$ \nabla_\theta J = \mathbb{E}[ \nabla_\theta \log \pi_\theta(a|s) (R_t - b(s)) ] $$


### Policy Gradient Theorem
$$\nabla_\theta J = \mathbb{E}[ \nabla_\theta \log \pi_\theta(a|s) A^{\pi}(s,a) ] $$


###  Natural Policy Gradient update
$$ \Delta \theta = \eta G^{-1}(\theta) \nabla_\theta J$$


---

## 💡 Conceptual Takeaway

> Policy Gradient methods directly optimize **expected reward** by following the gradient in **policy space**, not value space.  
> They serve as the foundation for **nearly all modern deep reinforcement learning algorithms** that rely on differentiable policies.

---
