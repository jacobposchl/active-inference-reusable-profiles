# K-Sweep Redesign: Context-Specific Specialization

## The Problem

The original K-sweep had K=2 configurations where:
- **Both profiles were neutral** on arm preferences (xi_left=0, xi_right=0)
- Profiles varied on gamma (precision), phi_loss (risk), xi_hint (hint-seeking)
- Z matrix routing was **arbitrary** - no alignment between profile characteristics and contexts

Result: **K=1 beat K=2** because the best exploratory profile (γ=2.0) worked well everywhere, and mixing it with exploitative profiles only made things worse.

## The Solution: Implement Advisor's Recommendations

### Core Insight

> "For the classic two-armed bandit k-sweep, your profiles should probably encode **context-relevant preferences** (C vectors aligned with which arm is better)."

The Z matrix routes profiles based on **context beliefs** (left_better vs right_better). Profiles should be **specialized for those contexts**.

### New K=1 Design: Context-Agnostic Baseline

**9 configurations** (reduced from 27):

```python
gamma: [2.0, 4.5, 7.0]           # Exploration vs exploitation
phi_loss: [-4.0, -7.0, -10.0]    # Risk aversion
xi_left: 0.0                      # NEUTRAL - no left bias
xi_right: 0.0                     # NEUTRAL - no right bias
xi_hint: 0.0                      # NEUTRAL - removed hint variation
```

**Key property:** K=1 profiles are **neutral** - they work equally regardless of context. Like M1 model, they can't adapt preferences based on beliefs.

### New K=2 Design: Context-Specialized Pairs

**15 configurations** (5 pairs × 3 Z matrices):

#### Pair 1: Pure Arm Specialists (Cleanest Test)
```python
Profile 1 (LEFT-specialist):
  gamma: 4.5
  phi_logits: [0.0, -7.0, 7.0]
  xi_logits: [0.0, 0.0, +3.0, -3.0]  # Favor left, avoid right

Profile 2 (RIGHT-specialist):
  gamma: 4.5
  phi_logits: [0.0, -7.0, 7.0]
  xi_logits: [0.0, 0.0, -3.0, +3.0]  # Avoid left, favor right

Z = [[1.0, 0.0],  # left_better → Profile 1 (left-specialist)
     [0.0, 1.0]]  # right_better → Profile 2 (right-specialist)
```

**Why this should win:**
- When agent believes "left is better" → activates left-specialist
- That profile **already prefers left arm** → faster decisions
- Better **inference coherence** (higher log-likelihood)
- Smooth mixing during uncertainty (after reversals)

#### Pairs 2-5: Additional Variations

2. **Arm specialists + precision variation** (left-exploit γ=7.0, right-explore γ=2.0)
3. **Arm specialists + loss aversion variation** (left-cautious φ=-10, right-bold φ=-4)
4. **Strong arm specialists** (xi biases of ±5 instead of ±3)
5. **Combined variation** (arm + precision + risk all varying)

Each tested with 3 Z configurations:
- Hard: `[[1,0], [0,1]]` - strict specialization
- Soft: `[[0.8,0.2], [0.2,0.8]]` - mostly specialized
- Balanced: `[[0.5,0.5], [0.5,0.5]]` - always equal mix (control)

## The Hypothesis

**K=2 should outperform K=1 because:**

1. **Alignment:** Profile structure matches task structure (2 contexts → 2 profiles)
2. **Specialization:** Each profile optimized for its assigned context
3. **Coherence:** Action preferences align with beliefs → better log-likelihood
4. **Adaptation:** Smooth mixing during uncertainty, strong commitment when confident

**Measured by:**
- Total rewards (primary metric)
- **Log-likelihood** (inference coherence - should be higher for K=2)
- Variance (K=2 should be more robust/consistent)

## What We're Actually Testing

**Research question:** "Does context-specific profile specialization provide benefits over a single global profile?"

**K=1:** Global neutral strategy
- Must work equally well for both "left better" and "right better" contexts
- Like having one tool for all jobs

**K=2:** Context-specialized strategies
- Each profile optimized for one context
- Z matrix provides belief-weighted switching
- Like having the right tool for each job

**K=3:** Testing redundancy handling
- More profiles than contexts (3 > 2)
- Tests whether framework handles gracefully
- Might discover sub-modes or waste resources

## Expected Outcomes

### If K=2 Wins (Validates Core Claim)
```
Best K=2 > Best K=1:
  - Context specialization provides measurable benefit
  - Profile differentiation matters
  - Z matrix routing is effective
```

### If K=1 Still Wins (Reveals Design Issues)
```
Possible reasons:
1. Arm biases too strong (overconstrain agent)
2. Task too simple (one good strategy beats specialization)
3. Profile mixing overhead outweighs benefits
4. Z assignment not optimal for this task structure
```

## Comparison to Old Design

### Old K=2 (BROKEN)
```python
Profile 0: γ=7.0, xi=[0, -1, 0, 0]  # Exploitative, avoid hints, NEUTRAL arms
Profile 1: γ=2.0, xi=[0, +2, 0, 0]  # Exploratory, seek hints, NEUTRAL arms

Z routes based on context but profiles don't specialize → arbitrary assignment
```

**Problem:** No alignment between profile characteristics and contexts they're routed to.

### New K=2 (FIXED)
```python
Profile 0: γ=4.5, xi=[0, 0, +3, -3]  # LEFT-specialist
Profile 1: γ=4.5, xi=[0, 0, -3, +3]  # RIGHT-specialist

Z routes based on context AND profiles ARE specialized → meaningful assignment
```

**Solution:** Profile structure aligns with context structure.

## Mathematical Advantage

### K=1 Expected Free Energy
```
EFE(π) = -E[log P(o|π)] - E[log Q(s|π)] + preferences

For "choose left" policy:
  preferences = 0.0 (neutral on both arms)
  
Agent must rely purely on state inference
```

### K=2 Expected Free Energy
```
When beliefs say "left_better" (80% confidence):
  Active profile has xi_left = +3.0
  
For "choose left" policy:
  preferences = +3.0 (BONUS for left)
  
For "choose right" policy:
  preferences = -3.0 (PENALTY for right)

Left policy preferred EVEN BEFORE state inference!
Beliefs + preferences REINFORCE each other
```

**Result:** Faster convergence, stronger commitment, better coherence.

## Implementation Verified

✅ K=1: 9 profiles, all ARM-NEUTRAL (xi_left=0, xi_right=0)  
✅ K=2: 15 configurations, all have ARM SPECIALIZATION  
✅ Z matrices: 3 types (hard, soft, balanced)  
✅ Profile descriptions updated to show arm preferences  
✅ Proper routing: left-specialist → left_better context

Ready to test!
