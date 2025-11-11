# ⏱️ Options in Reinforcement Learning (Temporal Abstraction & Hierarchical RL)

---

## 📌 What Are Options?

Options are a framework for introducing **temporal abstraction** into reinforcement learning. They allow an agent to act at multiple **timescales** by using **macro-actions** — sequences of primitive actions bundled into a single high-level action.

---

## 🔍 Motivation: Why Use Options?

- Standard RL policies operate at the level of **single-step actions**.
- Many tasks have **repetitive sub-tasks** or **long-term goals** (e.g., "walk to door", "pick up object").
- Options allow agents to:
  - Learn **structured behavior**
  - Reuse useful **subroutines**
  - Reduce decision frequency
  - Speed up learning in sparse-reward environments

---

## ⚙️ Formal Definition of an Option

An **option** is a triple:


$$\mathcal{O} = (\mathcal{I}, \pi, \beta)$$


Where:

| Symbol | Meaning |
|--------|---------|
| $$ \mathcal{I} $$ | **Initiation set**: states where the option can be started |
| $$ \pi $$ | **Intra-option policy**: what to do while the option is active |
| $$ \beta(s) $$ | **Termination condition**: probability of stopping in state \(s\) |

---

## 🧠 Execution Cycle of Options

1. At time \( t \), agent selects option \( o \) from available options (based on current state).
2. Executes the intra-option policy \( \pi \) until termination condition \( \beta \) is met.
3. Repeats with a new option.

> From the agent's perspective, options are **actions with variable durations**.

---

## ⏱️ Semi-Markov Decision Processes (SMDPs)

- Because options may last **multiple time steps**, the underlying process becomes a **semi-Markov decision process**.
- This generalizes the MDP framework by allowing actions (options) to take varying amounts of time.

---

## 📚 Learning with Options

There are two main tasks:

### 1. **Learning the Option Policies (\(\pi\))**

- Use standard RL methods within each option.
- Learn how to accomplish the subtask (e.g., walking, navigating).

### 2. **Learning the Meta-Policy (\(\mu\))**

- This is the **high-level controller** that decides **which option** to execute in a given state.
- Can be learned using Q-learning or policy gradients over options.

---

## 🎯 Benefits of Options

| Benefit | Description |
|--------|-------------|
| **Reusability** | Options like "navigate" or "open door" can be reused across tasks |
| **Hierarchical Planning** | Makes long-horizon tasks tractable |
| **Speed of Learning** | Reduces effective horizon for learning high-level goals |
| **Better Exploration** | Options explore structured sequences, not just single steps |
| **Interpretability** | Behavior can be broken into meaningful sub-behaviors |

---

## 🤖 Options in Modern Deep RL

### Hierarchical Reinforcement Learning (HRL)

- Combines **deep learning** and **options framework**.
- Usually involves two networks:
  1. **High-level policy (meta-controller)**: selects option or sub-goal
  2. **Low-level policy**: executes primitive actions for that option

---

### Notable HRL Algorithms Using Options

| Algorithm | Description |
|----------|-------------|
| **Option-Critic Architecture** | Learn intra-option policies and termination functions end-to-end using gradients |
| **FeUdal Networks** | Uses managers and workers; manager sets goals, worker fulfills them |
| **HIRO (Hierarchical Reinforcement Learning with Off-policy correction)** | Learns over sub-goals with relabeling to stabilize learning |
| **SNN4HRL (Stochastic Neural Networks for HRL)** | Uses discrete latent variables to select among sub-policies (like options) |

---

## 🧪 Examples of Option Use

1. **Maze Navigation**:
   - Options: "go to hallway", "open door", "go to goal"
2. **Robotics**:
   - Options: "pick up", "move arm to location", "release"
3. **Atari**:
   - Option discovery helps solve sparse-reward games (like Montezuma's Revenge)

---

## 🔄 Discovering Options Automatically

Instead of manually defining options, modern approaches **learn them** from data:

- **Bottleneck States**: Identify commonly visited transitions (e.g., doorways)
- **Subgoal Discovery**: Use unsupervised learning to find latent structure
- **Diversity-Driven Options**: Encourage diverse behaviors (via mutual information or contrastive loss)

---

## 🧠 Related Concepts

| Concept | Connection to Options |
|--------|------------------------|
| **Macro-actions** | Fixed action sequences (subset of options) |
| **Subgoals** | Can be learned or manually defined |
| **Temporal Abstraction** | The core principle behind options |
| **Feudal RL** | Hierarchical policy with managers setting goals |
| **Curriculum Learning** | Use simpler options to tackle more complex tasks |

---

## ✅ Summary

| Concept | Role in RL |
|--------|-------------|
| Option (\(\mathcal{I}, \pi, \beta\)) | Temporally extended action |
| Meta-policy | Chooses which option to execute |
| Intra-option policy | Executes primitive actions |
| SMDP | Underlying learning framework |
| Option-Critic | End-to-end learning of options |
| HRL | Scalable deep RL using hierarchy |

---

## 💡 Final Insight

> "Options enable agents to **think ahead** and **act intelligently over time**, not just in the moment. They're a stepping stone to scalable, structured, human-like RL."

---
