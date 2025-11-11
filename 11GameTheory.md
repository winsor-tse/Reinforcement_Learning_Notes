# 🎮 CS7642 – Lecture 11A-B-C: Game Theory

---

## 🤔 What Is Game Theory?

- Game Theory is the mathematics of **conflicting interests** in decision-making.
- It becomes relevant in **multi-agent environments**.
- Agents must reason about the decisions of others to act optimally.

---

## 🎲 A Simple Game

- Two agents, **A** and **B**, take alternating actions.
- The game is a **2-player, zero-sum, finite deterministic game with perfect information**.
- Rewards:
  - When A gains reward \( r \), B gets \( -r \).
- **Strategy** in Game Theory ≈ **Policy** in MDPs:
  - A strategy maps **all possible states** to actions.

### 📊 Matrix Representation

- The game can be summarized in a matrix of strategies (A vs B).
- Each cell shows the outcome for a given strategy pair.

---

## ♟️ Minimax Principle

- A and B aim to **minimize the other's maximum gain**.
- **Minimax Algorithm**:
  - A chooses the **max of minimums**.
  - B chooses the **min of maximums**.

### ✅ Key Result

- In 2-player, zero-sum, perfect-information games:
  \[
  \text{Minimax} = \text{Maximin}
  \]
- There always exists an **optimal pure strategy**.

---

## 🌲 Game Tree Extensions

### 1. Non-deterministic Game

- Some game states behave **stochastically** (random transitions).

### 2. Minipoker

- **Hidden information + stochasticity**.
- A receives a red (bad) or black (good) card (50% chance).
  - Red → Resign: A loses 20¢
  - Else hold:
    - B resigns → A gains 10¢
    - B sees card:
      - Red → A loses 40¢
      - Black → A gains 30¢

---

## 🎲 Mixed Strategies

- Instead of a fixed (pure) decision, use **probabilities** over strategies.
- Analyzed by treating one player as deterministic and the other probabilistic.
- **Maximin over outcome functions** determines value of the game.

---

## 🚨 The Snitch (Prisoner’s Dilemma)

- Two players choose to **Cooperate (C)** or **Defect (D)**.
- Best joint outcome: **both cooperate**
- Rational choice: **always defect**
  - Leads to **worse joint outcome** → (-6, -6)

---

## 🎯 Nash Equilibrium

A strategy profile 
$$ s^* = (s_1^*, s_2^*, \dots, s_n^*) $$ 
is a **Nash Equilibrium** if:

$$
s_i^* = \arg\max_{s_i} \text{Utility}_i(s_1^*, \dots, s_i, \dots, s_n^*)
$$

- No player has incentive to unilaterally change their strategy.
- Applies to **pure and mixed** strategies.

### 📚 Theorems

1. If strict dominance eliminates all but one strategy combo → it is the Nash Equilibrium.
2. Any Nash Equilibrium **survives strict dominance elimination**.
3. In any finite game → **at least one Nash Equilibrium exists**.

---

## 👣 The Two-Step & Iterated Prisoner’s Dilemma (IPD)

### 🔄 Finite Repetition:

- Backward induction → always defect in final step.
- Therefore, rational strategy = **defect in all steps**.

### ❓ What if number of rounds is **unknown**?

- Let continuation probability = \( \gamma \)
- Expected number of rounds:
  $$
  \frac{1}{1 - \gamma}
  $$
- Behaves like a **discount factor**.

### 🤝 Tit-for-Tat (TFT)

- Cooperate first, then copy opponent’s last move.

#### 📈 Strategy Payoffs:

| Strategy | Total Reward |
|----------|--------------|
| Always Defect | $$ \frac{-6\gamma}{1 - \gamma} $$ |
| Always Cooperate | $$ \frac{-1}{1 - \gamma} $$ |

Equate both:
\[
\frac{-6\gamma}{1 - \gamma} = \frac{-1}{1 - \gamma} \Rightarrow \gamma = \frac{1}{6}
\]

---

## 🧠 Best Response to Finite-State Strategy

- In multi-round games, actions affect **future reactions**.
- Use a **state machine**
