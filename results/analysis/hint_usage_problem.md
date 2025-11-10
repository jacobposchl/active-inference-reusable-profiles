# The Hint Usage Problem: Why Agents Ignore Hints

## Executive Summary

We're trying to create an environment where multi-profile agents outperform single-profile agents by using different strategies in different regimes. Specifically:
- **Stable regime**: Ignore hints, exploit directly
- **Volatile regime**: Use hints strategically (30-60% of the time)

**Current Status**: ❌ FAILING - Optimized agents use 0% hints in both regimes, achieving 297.9 total rewards by relying purely on active inference.

---

## The Core Problem: Active Inference Is Too Good

### What We Expected
In volatile regime with harsh penalties for wrong choices, agents should use hints to avoid catastrophic losses.

### What Actually Happens
**K=1 Optimized Agent Performance:**
- Total rewards: 297.9 / 400 trials (74.5% rate)
- Stable hint usage: 0.0%
- Volatile hint usage: 0.0% ⚠️
- Volatile reward rate: 54.4%

**How is this possible?**

The agent uses **Bayesian belief updating** to learn which arm is better through trial and error:

1. **Trial 1-2**: Agent guesses randomly, ~50% accuracy, some penalties
2. **Trial 3-8**: Agent has learned from feedback, ~70% accuracy, few penalties  
3. **Trial 9**: Reversal happens, back to trial 1

**Average performance**: ~54% accuracy across the reversal cycle, better than random (50%)!

---

## Why Hints Don't Help (Current Setup)

### Environment Parameters (Volatile Regime)
- Hint accuracy: 95%
- Reward probability: 65%
- Penalty for wrong choice: -5.0
- Reversals: Every 8 trials

### Expected Value Calculations

**Active Inference Strategy** (no hints):
- Across 8-trial cycle: ~54% accuracy
- EV = 0.54×0.65 + 0.46×(-5.0) = 0.351 - 2.3 = **-1.95 per trial**
- But over 8 trials: early penalties, late rewards = net positive!

**Hint Strategy** (take hint then choose):
- Trial 1: Take hint → 0 reward
- Trial 2: Make hint-guided choice → 0.95×0.65 = 0.6175 reward
- **Average: 0.309 reward per trial**

Wait... hints SHOULD be better! But they're not being used. Why?

### The Opportunity Cost Problem

**Key Issue**: Hints cost an entire trial (action slot) but give 0 immediate reward.

From the optimizer's perspective:
- Taking a hint = guaranteed 0 reward this trial
- Making a choice = potential for 0.65 reward this trial (and info for next trial)

The agent values **exploration through choosing** more than **exploration through hinting** because choosing gives:
1. Potential reward (0.65 if correct)
2. Information about which arm is better (for future trials)
3. No opportunity cost (already making a choice anyway)

Hints only give information, no reward, AND cost the opportunity to earn reward that trial.

---

## Failed Solutions Attempted

### Attempt 1: Increase Penalty
- **Tried**: penalty = -2.0
- **Result**: Still 0% hints (agent learns fast enough to avoid most penalties)

### Attempt 2: Much Harsher Penalty  
- **Tried**: penalty = -5.0
- **Result**: Still 0% hints (54.4% accuracy sufficient to net positive)

### Attempt 3: Penalize All Direct Choices
- **Tried**: penalty = -5.0 for wrong, -1.0 for any direct choice in volatile
- **Result**: Hint-spamming! Agent uses 69% hints and barely chooses (only 108.5 total rewards)
- **Why**: Even correct hint-guided choices lose money (0.6175 - 1.0 = -0.38)

### Attempt 4: Faster Reversals
- **Tried**: Reversals every 2 trials instead of 8
- **Problem Identified**: Hints become stale before you can use them!
  - Trial 1: Take hint
  - Trial 2: Reversal happens, hint is now WRONG
  - Making choice on stale hint = worse than guessing

---

## The Fundamental Issue

**Hints have an opportunity cost that rewards don't.**

When you take a hint:
- You get: Information about current state
- You lose: Chance to earn reward AND information (from reward feedback)

When you make a choice:
- You get: Potential reward + feedback about whether you were right
- You lose: Nothing (this was your action anyway)

**The trade-off only makes sense if:**
1. Hints are free (separate observation modality, not an action), OR
2. Direct choices are so punishing that the risk outweighs the reward, OR  
3. The task is literally impossible to learn without hints (no reward feedback)

Active inference is fundamentally designed to learn through action, so it will always prefer actions that give both reward potential AND information over actions that only give information.

---

## Potential Solutions (Not Yet Implemented)

### Option 1: Make Hints Free Observations
Instead of hint being an action (costs a trial), make it a separate observation modality:
- Agent can see hint alongside choice observation
- No opportunity cost
- Problem: Requires major architectural change to environment

### Option 2: No-Feedback Regime
In volatile regime, don't show reward/loss outcomes - only hints reveal truth:
- Makes active inference impossible
- Hints become only source of information
- Problem: Deviates significantly from typical RL setup

### Option 3: Extremely Fast Random Reversals
Random reversals with high probability (e.g., 30% chance per trial):
- Agent can never rely on learned beliefs
- Must constantly re-check with hints
- Problem: Unclear if this creates interpretable regime-specific strategies

### Option 4: Multi-Step Policies (Best Candidate?)
Allow policies like "take hint THEN make choice":
- 2-step policy: [hint, left] or [hint, right]
- Cost: 2 trials, Benefit: 0.95×0.65 = 0.6175 reward
- Compare to: [left, left] = 2 trials, ~1.08 reward via active inference
- Problem: Still worse than pure active inference!

### Option 5: Accept That Hints Are Suboptimal
Maybe the real lesson is that in this task structure, **active inference through choosing is genuinely better than using hints**, and we need a fundamentally different task where hints are actually necessary (e.g., perceptual uncertainty, hidden variables, etc.)

---

## Current Best Performance

**Hand-Crafted K=2 (with oracle regime knowledge):**
- Total rewards: 219.2 / 400 trials
- Stable hints: 0.0%
- Volatile hints: 27.2%
- Beats K=1 by +26.8 rewards

But optimized K=1 WITHOUT regime knowledge gets 297.9 rewards by ignoring hints entirely!

---

## Next Steps / Open Questions

1. **Is the task fundamentally flawed?** Maybe hints can't compete with active inference in this setup.

2. **Should we redesign the task?** What kind of task actually REQUIRES hints to be optimal?

3. **Is regime-specificity achievable?** Can we create environments where different profiles genuinely help, or does active inference smooth over regime differences?

4. **What's the real contribution?** If K=1 active inference already solves everything, what value do multi-profiles add?

---

## Key Insight

**Active inference agents are Bayesian learners that naturally balance exploration and exploitation through action.** Any task where "gather information through action" is viable will naturally favor direct choices over costly information-seeking actions like hints.

To make hints useful, we need tasks where:
- Actions don't provide learning signal (no feedback)
- Uncertainty is too high for action-based learning (extremely volatile)
- Hints provide information that's literally unavailable through action

Otherwise, the agent will always prefer to learn by doing rather than asking.
