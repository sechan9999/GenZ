# Chapter 1 — Section 4: Recursion & Backtracking

## Core Concepts

**Recursion** solves a problem by reducing it to smaller versions of itself. Every recursive solution needs:
1. **Base case** — when to stop
2. **Recursive case** — make the problem smaller and call yourself

**Backtracking** is recursion with "undo" — you explore a choice, and if it doesn't lead to a solution, you undo it (backtrack) and try the next option. Think of it as DFS over a decision tree.

**The backtracking template:**
```python
def backtrack(state, choices):
    if is_solution(state):
        result.append(copy(state))  # IMPORTANT: copy, not reference
        return
    for choice in choices:
        make(choice, state)
        backtrack(state, next_choices)
        undo(choice, state)          # backtrack
```

---

## Q&A 1 — Fibonacci (Classic Recursion)

**Q:** Compute the n-th Fibonacci number. What's the time complexity of the naive approach?

**A — Naive: O(2^n) — each call spawns two more:**
```python
def fib_naive(n: int) -> int:
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
```

**A — With memoization: O(n) time, O(n) space:**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    if n <= 1:
        return n
    return fib_memo(n-1) + fib_memo(n-2)

# Or explicit cache
def fib_memo_explicit(n: int, memo={}) -> int:
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo_explicit(n-1, memo) + fib_memo_explicit(n-2, memo)
    return memo[n]
```

**A — Bottom-up DP: O(n) time, O(1) space:**
```python
def fib(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

**Say the progression:** "Naive is 2^n due to overlapping subproblems. Memoization caches results and brings it to O(n). Bottom-up DP avoids recursion overhead and uses O(1) space."

---

## Q&A 2 — Power of a Number (Recursion Structure)

**Q:** Implement `pow(x, n)` — raise x to the power n. What is the optimal approach?

**A — Naive O(n) is too slow. Use fast exponentiation O(log n):**
```python
def my_pow(x: float, n: int) -> float:
    if n < 0:
        x, n = 1/x, -n
    if n == 0:
        return 1
    if n % 2 == 0:
        half = my_pow(x, n // 2)
        return half * half          # don't call twice!
    else:
        return x * my_pow(x, n - 1)

# Example
print(my_pow(2.0, 10))   # 1024.0
print(my_pow(2.0, -2))   # 0.25
```

**Key insight:** "x^n = (x^(n/2))² when n is even. This halves the problem each time — O(log n). The common mistake is writing `my_pow(x, n//2) * my_pow(x, n//2)` which makes two recursive calls and loses the speedup."

---

## Q&A 3 — Generate Parentheses (Backtracking)

**Q:** Generate all valid combinations of `n` pairs of parentheses.

**A — Classic backtracking:**
```python
def generate_parentheses(n: int) -> list[str]:
    result = []

    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result

# Example
print(generate_parentheses(3))
# ['((()))', '(()())', '(())()', '()(())', '()()()']
```

**Complexity:** O(4^n / √n) — the n-th Catalan number.

**Explain the validity rules:**
> "I can add `(` whenever I haven't used all n. I can add `)` only when there are more open than close brackets. This pruning ensures every string in the result is valid."

---

## Q&A 4 — Climbing Stairs (Recursion → DP Bridge)

**Q:** You can climb 1 or 2 steps at a time. How many distinct ways can you climb n stairs?

**A:**
```python
# This IS Fibonacci in disguise
def climb_stairs(n: int) -> int:
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n+1):
        a, b = b, a + b
    return b

# Example
print(climb_stairs(5))  # 8
```

**Show you see the pattern:**
> "To reach step n, I could have come from step n-1 (1 step) or step n-2 (2 steps). So `ways(n) = ways(n-1) + ways(n-2)`. That's Fibonacci. Base cases: 1 way to reach step 1, 2 ways to reach step 2."

---

## Q&A 5 — Letter Combinations of a Phone Number

**Q:** Given a string of digits 2–9, return all letter combinations the digits could represent (like a phone keypad).

**A — Backtracking over digits:**
```python
def letter_combinations(digits: str) -> list[str]:
    if not digits:
        return []

    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    result = []

    def backtrack(index, current):
        if index == len(digits):
            result.append(current)
            return
        for letter in phone[digits[index]]:
            backtrack(index + 1, current + letter)

    backtrack(0, '')
    return result

# Example
print(letter_combinations("23"))
# ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

**Complexity:** O(4^n · n) where n is number of digits (4 because some digits have 4 letters)

---

## Q&A 6 — Interview Dialogue: Stack Overflow Risk

**Interviewer:** "What happens if n is 100,000 in your recursive Fibonacci?"

**You:** "Python's default recursion limit is 1,000. It would raise a `RecursionError`. Two fixes: use `sys.setrecursionlimit()` (hacky), or convert to an iterative bottom-up DP approach — which I'd prefer. Iterative avoids call stack overhead entirely and is O(1) space for Fibonacci."

---

## Backtracking Decision Tree — How to Think About It

```
Problem: Generate all subsets of [1, 2, 3]

                 []
          /      |      \
        [1]     [2]     [3]
       /   \      \
    [1,2] [1,3]  [2,3]
      |
  [1,2,3]

At each node: include current element or skip it.
When you've considered all elements: record the state.
```

**Template for subset-style backtracking:**
```python
def subsets(nums):
    result = []
    def backtrack(start, current):
        result.append(list(current))    # every state is a valid subset
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()               # undo
    backtrack(0, [])
    return result
```

---

## Pattern Recognition Cheat Sheet

| Signal | Recursion / Backtracking pattern |
|---|---|
| "generate all / count all ways" | Backtracking |
| "optimal substructure" | Recursion + memoization → DP |
| "tree structure / nested" | Natural recursion |
| "undo a choice" | Backtrack with explicit undo |
| "overlapping subproblems" | Add memoization |

---

## Common Mistakes

1. **Missing base case** — infinite recursion; always handle `n == 0` or empty input
2. **Appending a reference, not a copy** — `result.append(current)` appends a reference that later gets mutated; do `result.append(list(current))` or `result.append(current[:])`
3. **Forgetting to undo** — if you mutate state before recursing, undo it after; immutable approach (`s + '('`) avoids this
4. **Exponential without memoization** — if you see overlapping subproblems, cache results
