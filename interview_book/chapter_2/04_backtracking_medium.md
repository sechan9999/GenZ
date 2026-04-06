# Chapter 2 — Pattern 4: Backtracking (Medium Problems)

## The Pattern Revisited

Backtracking = DFS on a decision tree, with pruning.

**The universal template:**
```python
def backtrack(start, current_state):
    if base_case_reached():
        result.append(copy_of(current_state))
        return

    for choice in get_choices(start):
        if is_valid(choice):              # PRUNE here
            make_choice(choice)
            backtrack(next_start, current_state)
            undo_choice(choice)           # BACKTRACK
```

**Three core questions to answer before coding:**
1. What is the choice at each step?
2. When do I record a result (vs when do I prune)?
3. What is the start index / constraint to avoid duplicates?

---

## Q&A 1 — Subsets

**Q:** Return all possible subsets of a distinct integer array (the power set).

**A — Include/exclude at each index:**

```python
def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start, current):
        result.append(list(current))          # every state is a valid subset

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)         # i+1: don't reuse elements
            current.pop()

    backtrack(0, [])
    return result

# Example
print(subsets([1,2,3]))
# [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
```

**Complexity:** Time O(n · 2^n), Space O(n) stack depth

**Explain why we record at every step:** "Every node in the decision tree is a valid subset — the empty set, partial subsets, the full set. So I record before deciding whether to include more elements."

---

## Q&A 2 — Permutations

**Q:** Return all permutations of a distinct integer array.

**A — Swap-based or used-set approach:**

```python
# Approach 1: Track used elements with a boolean array
def permute(nums: list[int]) -> list[list[int]]:
    result = []
    used = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            result.append(list(current))
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                current.append(nums[i])
                backtrack(current)
                current.pop()
                used[i] = False

    backtrack([])
    return result

# Approach 2: Swap in-place
def permute_swap(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(list(nums))
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]   # undo

    backtrack(0)
    return result

# Example
print(permute([1,2,3]))
# [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Complexity:** Time O(n · n!), Space O(n)

**Difference from subsets:** "Permutations care about order. I don't pass a `start` index — I pick any unused element at each position. The `used` array prevents re-picking."

---

## Q&A 3 — Combination Sum

**Q:** Given candidates (distinct positive integers) and a target, return all unique combinations that sum to target. Each number can be used unlimited times.

**A — Allow re-use by not incrementing start:**

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result = []
    candidates.sort()   # enables pruning

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(list(current))
            return
        for i in range(start, len(candidates)):
            c = candidates[i]
            if c > remaining:       # PRUNE: sorted, so all further are bigger
                break
            current.append(c)
            backtrack(i, current, remaining - c)   # i (not i+1): allow reuse
            current.pop()

    backtrack(0, [], target)
    return result

# Example
print(combination_sum([2,3,6,7], 7))   # [[2,2,3],[7]]
print(combination_sum([2,3,5], 8))     # [[2,2,2,2],[2,3,3],[3,5]]
```

**Complexity:** Time O(N^(T/M)) where T=target, M=min candidate. Space O(T/M) stack depth.

**Key decisions:**
- Sort → enables early termination (`break` when `c > remaining`)
- Pass `i` not `i+1` → allows reuse of same element

---

## Q&A 4 — Word Search

**Q:** Given a 2D grid of characters, check if a given word exists by following adjacent cells (no reuse).

**A — DFS with backtracking on the visited set:**

```python
def exist(board: list[list[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])
    DIRS = [(0,1),(0,-1),(1,0),(-1,0)]

    def dfs(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[idx]:
            return False

        temp = board[r][c]
        board[r][c] = '#'           # mark visited (backtrack: restore after)

        found = any(dfs(r+dr, c+dc, idx+1) for dr, dc in DIRS)

        board[r][c] = temp          # BACKTRACK: restore
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False

# Example
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
print(exist(board, "ABCCED"))  # True
print(exist(board, "SEE"))     # True
print(exist(board, "ABCB"))    # False  (can't revisit B)
```

**Complexity:** Time O(m·n·4^L) where L=word length, Space O(L) stack

**The backtrack step:** "I mark the cell as visited by overwriting with '#'. After recursion returns — whether it succeeded or not — I restore the original character. This is classic backtracking: try, recurse, undo."

---

## Q&A 5 — N-Queens

**Q:** Place n queens on an n×n chessboard so no two queens attack each other. Return all solutions.

**A — Place one queen per row, track columns and diagonals:**

```python
def solve_n_queens(n: int) -> list[list[str]]:
    result = []
    cols = set()
    pos_diag = set()    # r - c is constant on / diagonals
    neg_diag = set()    # r + c is constant on \ diagonals

    board = [['.' for _ in range(n)] for _ in range(n)]

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in pos_diag or (row + col) in neg_diag:
                continue
            # Place queen
            cols.add(col)
            pos_diag.add(row - col)
            neg_diag.add(row + col)
            board[row][col] = 'Q'

            backtrack(row + 1)

            # Remove queen (backtrack)
            cols.remove(col)
            pos_diag.remove(row - col)
            neg_diag.remove(row + col)
            board[row][col] = '.'

    backtrack(0)
    return result

# Example
print(len(solve_n_queens(4)))   # 2 solutions
print(len(solve_n_queens(8)))   # 92 solutions
```

**Complexity:** O(n!) time (very roughly), O(n) space for the tracking sets

**Explain the diagonal math:** "Two queens are on the same '/' diagonal if `row - col` is equal. They're on the same '\\' diagonal if `row + col` is equal. By tracking these sets, I check attack conflicts in O(1)."

---

## Q&A 6 — Interview Dialogue: Optimizing Backtracking

**Interviewer:** "Your solution works but is slow. How would you optimize it?"

**You:** "Backtracking optimization is almost always about pruning — cutting branches early before recursing into them.

Key pruning strategies:
1. **Sort candidates** and break early when a candidate exceeds remaining budget (Combination Sum)
2. **Constraint propagation** — in N-Queens, track which columns/diagonals are taken in O(1) sets rather than scanning the board
3. **Symmetry breaking** — for N-Queens, only place in the first half of row 0 and double the count (advanced)
4. **Bitmask instead of set** — for small n, use integer bits to track used columns/diagonals. O(1) operations, better cache performance

In an interview, I'd code the clean version first, then optimize with pruning as the interviewer asks."

---

## Pattern Recognition Cheat Sheet

| Problem | Key decisions |
|---|---|
| Subsets | Record at every node; pass `start+1` |
| Permutations | No `start` index; use `visited` set |
| Combinations | Record at leaf; pass `start` or `start+1` |
| Combination with reuse | Pass `start` (not `+1`) |
| Grid word search | Mark visited by mutating; restore after |
| N-Queens | One per row; track cols + diagonals |

---

## Common Mistakes

1. **Appending current without copying** — `result.append(current)` appends a reference. Always `result.append(list(current))` or `result.append(current[:])`
2. **Forgetting to undo** — every `make_choice(choice)` must have a matching `undo_choice(choice)` after the recursive call
3. **Wrong `start` parameter** — not passing `start` → infinite recursion / duplicates; passing `start+1` when reuse allowed → misses valid combos
4. **Not pruning sorted candidates** — forgetting `break` (using `continue`) after a candidate exceeds the remaining sum
