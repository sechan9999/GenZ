# Chapter 1 — Section 6: Dynamic Programming (1D Patterns)

## Core Concepts

**Dynamic programming** = recursion + memoization, or equivalently, filling a table bottom-up.

**Two signals that DP is the right tool:**
1. **Optimal substructure** — optimal answer to the whole problem uses optimal answers to subproblems
2. **Overlapping subproblems** — the same subproblem is computed multiple times in recursion

**The two approaches:**
- **Top-down (memoization):** Write the recursive solution; cache results
- **Bottom-up (tabulation):** Compute from smallest subproblems up to the answer

**The DP process:**
1. Define what `dp[i]` means in plain English
2. Write the recurrence relation
3. Identify base cases
4. Determine traversal order (usually left to right for 1D)
5. Optimize space if possible

---

## Q&A 1 — Climbing Stairs

**Q:** You can take 1 or 2 steps. How many ways to climb n stairs?

**A — Define, recur, optimize:**

```python
# Step 1: Define dp[i] = number of ways to reach stair i
# Step 2: Recurrence: dp[i] = dp[i-1] + dp[i-2]
#         (came from i-1 in one step, or from i-2 in two steps)
# Step 3: Base cases: dp[1] = 1, dp[2] = 2

# Bottom-up table — O(n) time, O(n) space
def climb_stairs_table(n: int) -> int:
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Space-optimized — O(n) time, O(1) space
def climb_stairs(n: int) -> int:
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1

print(climb_stairs(5))   # 8
print(climb_stairs(10))  # 89
```

**Say the insight:** "I only ever need the previous two values, so I can drop the whole array and just track two variables."

---

## Q&A 2 — House Robber

**Q:** An array represents money in each house. You cannot rob adjacent houses. Find the maximum you can rob.

**A:**
```python
# dp[i] = max money robbing from houses 0..i
# Recurrence: dp[i] = max(dp[i-1],          # skip house i
#                         dp[i-2] + nums[i]) # rob house i
# Base cases: dp[0] = nums[0], dp[1] = max(nums[0], nums[1])

def rob(nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
    return prev1

print(rob([1,2,3,1]))         # 4  (rob houses 0 and 2)
print(rob([2,7,9,3,1]))       # 12 (rob houses 0, 2, 4)
```

**Complexity:** Time O(n), Space O(1)

**Talk through the recurrence:** "At each house I have two choices: skip it (take `dp[i-1]`) or rob it (take `nums[i] + dp[i-2]`, since I can't take `dp[i-1]`). The max of these two is `dp[i]`."

---

## Q&A 3 — Maximum Subarray (Kadane's Algorithm)

**Q:** Find the contiguous subarray with the largest sum.

**A:**
```python
# dp[i] = maximum subarray sum ending exactly at index i
# Recurrence: dp[i] = max(nums[i],              # start fresh
#                         dp[i-1] + nums[i])     # extend previous

def max_subarray(nums: list[int]) -> int:
    best = nums[0]
    current = nums[0]
    for n in nums[1:]:
        current = max(n, current + n)    # extend or restart
        best = max(best, current)
    return best

print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # 6  (subarray [4,-1,2,1])
print(max_subarray([1]))                        # 1
print(max_subarray([5,4,-1,7,8]))               # 23
```

**Complexity:** Time O(n), Space O(1)

**Explain Kadane's key idea:** "If the running sum goes negative, there's no point carrying it forward — a fresh start from the current element is always better. So I reset to `nums[i]` whenever `current + nums[i] < nums[i]`."

---

## Q&A 4 — Coin Change

**Q:** Given coins of certain denominations and a target amount, find the minimum number of coins to make that amount. Return -1 if impossible.

**A — Classic unbounded knapsack:**
```python
# dp[i] = minimum coins needed to make amount i
# Recurrence: dp[i] = min(dp[i - coin] + 1) for each coin <= i
# Base case: dp[0] = 0 (zero coins to make amount 0)

def coin_change(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1,5,10,25], 36))   # 3 (25+10+1)
print(coin_change([1,5,10,25], 0))    # 0
print(coin_change([2], 3))            # -1
```

**Complexity:** Time O(amount × |coins|), Space O(amount)

**Walk through the table:** "I fill dp from 0 to amount. For each amount i, I try every coin. If I can make `i - coin` and add one more coin, that's a candidate for `dp[i]`."

---

## Q&A 5 — Longest Increasing Subsequence

**Q:** Find the length of the longest strictly increasing subsequence.

**A — O(n²) DP (explain first, optimize after):**
```python
# dp[i] = length of LIS ending at index i
# Recurrence: dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
# Base case: dp[i] = 1 (single element)

def length_of_lis_n2(nums: list[int]) -> int:
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# O(n log n) — patience sorting with binary search
from bisect import bisect_left

def length_of_lis(nums: list[int]) -> int:
    sub = []       # "piles" — not the actual LIS, just length tracking
    for n in nums:
        pos = bisect_left(sub, n)     # where would n go?
        if pos == len(sub):
            sub.append(n)             # extends LIS
        else:
            sub[pos] = n              # replaces to keep values as small as possible
    return len(sub)

print(length_of_lis([10,9,2,5,3,7,101,18]))  # 4  ([2,3,7,18] or [2,5,7,101])
print(length_of_lis([0,1,0,3,2,3]))           # 4
```

**Complexity:** O(n²) / O(n log n) | Space O(n)

---

## Q&A 6 — Interview Dialogue: Recognizing a DP Problem

**Interviewer:** "How do you know if a problem needs DP?"

**You:** "I look for two signals together. First, *optimal substructure* — can I express the best answer in terms of best answers to smaller versions? Second, *overlapping subproblems* — if I wrote a naive recursive solution, would it recompute the same state many times?

If yes to both, I start by writing the recursive solution, identify what state I need to track, and either memoize top-down or build a table bottom-up. Common DP shapes:

- **Linear scan** (1D): climbing stairs, house robber, max subarray
- **Two-sequence** (2D): edit distance, longest common subsequence
- **Knapsack**: coin change, subset sum
- **Interval**: palindrome partitioning, burst balloons"

---

## The 5-Step DP Problem-Solving Framework

Write this on the whiteboard as a guide:

```
1. DEFINE:    What does dp[i] represent? (Say it in plain English)
2. RECUR:     How does dp[i] relate to earlier states?
3. BASE:      What are the smallest/boundary values?
4. ORDER:     Left to right? Right to left? 2D diagonal?
5. OPTIMIZE:  Can I reduce from O(n) to O(1) space?
```

**Applied to House Robber:**
```
1. DEFINE:  dp[i] = max money robbing houses 0..i
2. RECUR:   dp[i] = max(dp[i-1], dp[i-2] + nums[i])
3. BASE:    dp[0] = nums[0]; dp[1] = max(nums[0], nums[1])
4. ORDER:   left to right
5. OPTIMIZE: only need last two values → use two variables
```

---

## Common Mistakes

1. **Wrong definition of dp[i]** — spend time defining it clearly; a bad definition derails everything
2. **Off-by-one in table size** — `dp = [0] * (n+1)` for amounts 0..n, not 0..n-1
3. **Initializing with 0 when 0 is a valid bad answer** — use `float('inf')` for min problems, `-float('inf')` for max
4. **Forgetting to return the answer from the right index** — `dp[n]`, `max(dp)`, or `dp[-1]` depending on definition
5. **Premature space optimization** — get the O(n) table right first, then optimize to O(1) if needed
