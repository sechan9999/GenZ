# Chapter 2 — Pattern 6: Dynamic Programming (Medium)

## Moving from 1D to Real DP

Chapter 1 covered 1D DP (single sequence). Here we tackle:
- **Knapsack variants** — choosing items with constraints
- **String DP** — two sequences, 2D table
- **Interval DP** — subproblems on a range `dp[i][j]`
- **State machine DP** — multiple states at each position

---

## Q&A 1 — Unique Paths

**Q:** A robot starts at the top-left of an `m × n` grid and can only move right or down. How many unique paths to reach the bottom-right?

**A — Classic 2D DP:**
```python
# dp[r][c] = number of ways to reach cell (r, c)
# Recurrence: dp[r][c] = dp[r-1][c] + dp[r][c-1]
# Base cases: first row and first column = 1 (only one way)

def unique_paths(m: int, n: int) -> int:
    dp = [[1] * n for _ in range(m)]

    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = dp[r-1][c] + dp[r][c-1]

    return dp[m-1][n-1]

# Space-optimized to O(n): only need the previous row
def unique_paths_optimized(m: int, n: int) -> int:
    row = [1] * n
    for _ in range(1, m):
        for c in range(1, n):
            row[c] += row[c-1]    # row[c] was prev row's value; row[c-1] is current row's left
    return row[n-1]

print(unique_paths(3, 7))   # 28
print(unique_paths(3, 2))   # 3
```

**Complexity:** Time O(m·n), Space O(n) optimized

---

## Q&A 2 — Coin Change II (Count Ways)

**Q:** Given coin denominations and an amount, count the number of distinct combinations that make that amount. (Contrast with Coin Change I which minimizes coins.)

**A — Unbounded knapsack, count combinations:**
```python
def change(amount: int, coins: list[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1   # one way to make amount 0: use nothing

    for coin in coins:              # outer: over coins
        for amt in range(coin, amount + 1):   # inner: over amounts
            dp[amt] += dp[amt - coin]

    return dp[amount]

# Example
print(change(5, [1,2,5]))    # 4  ([5], [2+2+1], [2+1+1+1], [1+1+1+1+1])
print(change(3, [2]))        # 0
print(change(10, [10]))      # 1
```

**Complexity:** Time O(amount × |coins|), Space O(amount)

**Why outer=coins, inner=amounts (not reversed)?** "Iterating coins in the outer loop ensures each coin is considered independently. If I reversed (outer=amounts, inner=coins), I'd count permutations (ordered) instead of combinations (unordered). Try `change(5, [1,2])` both ways to see the difference."

---

## Q&A 3 — Longest Common Subsequence

**Q:** Given two strings, find the length of their longest common subsequence (subsequence: characters in order, not necessarily adjacent).

**A — Classic 2D DP:**
```python
# dp[i][j] = LCS of s1[:i] and s2[:j]
# If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
# Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])

def longest_common_subsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# Example
print(longest_common_subsequence("abcde", "ace"))   # 3 ("ace")
print(longest_common_subsequence("abc", "abc"))     # 3
print(longest_common_subsequence("abc", "def"))     # 0
```

**Complexity:** Time O(m·n), Space O(m·n) → optimizable to O(n)

**Trace through small example on the whiteboard:**
```
      ""  a  c  e
   ""  0  0  0  0
   a   0  1  1  1
   b   0  1  1  1
   c   0  1  2  2
   d   0  1  2  2
   e   0  1  2  3   ← answer
```

---

## Q&A 4 — Best Time to Buy and Sell Stock with Cooldown

**Q:** After selling, you must wait one day before buying again (cooldown). Find max profit.

**A — State machine DP with three states:**

```python
def max_profit_cooldown(prices: list[int]) -> int:
    # States:
    # held:    currently holding a stock (best cash position)
    # sold:    just sold today (must cool down tomorrow)
    # rest:    in cooldown / not holding (ready to buy)

    held = -float('inf')     # can't hold without buying first
    sold = 0
    rest = 0

    for price in prices:
        prev_held = held
        prev_sold = sold
        prev_rest = rest

        held = max(prev_held,          # keep holding
                   prev_rest - price)  # buy today (only from rest state)
        sold = prev_held + price       # sell today
        rest = max(prev_rest,          # stay resting
                   prev_sold)          # cooldown expires, now resting

    return max(sold, rest)

# Example
print(max_profit_cooldown([1,2,3,0,2]))   # 3 (buy@1, sell@2, cool, buy@0, sell@2)
print(max_profit_cooldown([1]))           # 0
```

**Complexity:** Time O(n), Space O(1)

**State machine diagram:**
```
rest → (buy) → held → (sell) → sold
 ↑                               |
 └───────────── (cooldown) ──────┘
```

---

## Q&A 5 — Partition Equal Subset Sum

**Q:** Given an integer array, can you partition it into two subsets with equal sum?

**A — 0/1 Knapsack: can we reach `total/2`?**

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0:
        return False    # odd total can't be split equally

    target = total // 2

    # dp[j] = True if we can make sum j using some subset
    dp = [False] * (target + 1)
    dp[0] = True    # can always make sum 0 (empty subset)

    for num in nums:
        # Traverse right to left to avoid using num twice (0/1 knapsack)
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]

# Example
print(can_partition([1,5,11,5]))   # True  (1+5+5=11)
print(can_partition([1,2,3,5]))    # False
```

**Complexity:** Time O(n × target), Space O(target)

**The right-to-left traversal is key:** "In 0/1 knapsack, each item can be used at most once. By traversing `j` from right to left, when I compute `dp[j]`, `dp[j-num]` still reflects the state without `num` included. If I went left to right, I might include the same item twice."

---

## Q&A 6 — Decode Ways

**Q:** A string of digits maps to letters (1→A, 2→B, ..., 26→Z). Count the number of ways to decode it.

**A — 1D DP with two-digit lookahead:**
```python
def num_decodings(s: str) -> int:
    if not s or s[0] == '0':
        return 0

    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1          # empty string: one way
    dp[1] = 1          # single non-zero digit: one way

    for i in range(2, n + 1):
        one_digit = int(s[i-1])         # current digit alone
        two_digits = int(s[i-2:i])      # current + previous digit

        if one_digit != 0:
            dp[i] += dp[i-1]            # s[i-1] decodes alone

        if 10 <= two_digits <= 26:
            dp[i] += dp[i-2]            # s[i-2:i] decodes as a letter

    return dp[n]

# Example
print(num_decodings("12"))     # 2  ("AB" or "L")
print(num_decodings("226"))    # 3  ("BBF", "BZ", "VF")
print(num_decodings("06"))     # 0  (leading zero invalid)
```

**Complexity:** Time O(n), Space O(n) → O(1) with two variables

**Trace through "226":**
```
dp[0]=1, dp[1]=1 (for "2")
i=2 (char='2'): one='2'(ok) → dp[2]+=dp[1]=1; two='22'(ok,≤26) → dp[2]+=dp[0]=1; dp[2]=2
i=3 (char='6'): one='6'(ok) → dp[3]+=dp[2]=2; two='26'(ok,≤26) → dp[3]+=dp[1]=1; dp[3]=3
```

---

## Q&A 7 — Interview Dialogue: 1D vs 2D DP

**Interviewer:** "How do you decide if a DP problem needs a 1D or 2D table?"

**You:** "I count the dimensions of my state. One sequence with a single property → 1D. Two sequences → 2D (rows for one sequence, columns for the other, like LCS or Edit Distance). One sequence with two properties (e.g., index and remaining budget) → 2D. State machine with a small number of states → 1D table where I track multiple variables per position.

After I get the table right, I look at my recurrence and ask: does `dp[i]` depend on `dp[i-1]` only? Then I can compress to O(1). Does `dp[i][j]` depend only on `dp[i-1][j]` and `dp[i][j-1]`? Then I can compress the 2D table to a 1D rolling row."

---

## Pattern Recognition Cheat Sheet

| Problem | DP shape | State meaning |
|---|---|---|
| Grid paths | 2D | `dp[r][c]` = ways/cost to reach cell |
| LCS / Edit Distance | 2D | `dp[i][j]` = answer for prefix `s1[:i], s2[:j]` |
| 0/1 Knapsack | 1D (right-to-left) | `dp[j]` = can we reach weight/sum j |
| Unbounded Knapsack | 1D (left-to-right) | Same, but items reusable |
| State machine | Multiple 1D vars | `held, sold, rest` per position |
| Decode ways | 1D | `dp[i]` = decodings of `s[:i]` |

---

## Common Mistakes

1. **Wrong traversal order in knapsack** — 0/1 knapsack: right to left; unbounded: left to right. Getting this wrong gives wrong answers silently.
2. **Off-by-one in string DP** — use `dp[i]` for prefix `s[:i]` (length i), so the table has size `n+1`
3. **Not initializing dp[0] correctly** — `dp[0] = 1` for "ways to make 0" or "empty string has one decoding" problems
4. **Forgetting edge cases** — zero amounts, empty strings, single elements — handle before the DP loop
