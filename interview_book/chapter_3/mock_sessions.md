# Chapter 3 — Timed Mock Interview Sessions

## How to Use This Chapter

Each session simulates a real 45-minute technical screen. The format:
- **0–5 min:** Problem statement, clarification questions
- **5–10 min:** Brute force approach, discuss complexity
- **10–30 min:** Optimal solution, code it
- **30–40 min:** Test with examples, find bugs
- **40–45 min:** Follow-up questions from interviewer

**Rules for practice:**
1. Set a timer for 45 minutes
2. Talk out loud as you think — narrate everything
3. Write code first, then test it; don't run it in an IDE
4. After the timer, grade yourself using the scorecard below

---

## How to Talk Through a Problem (The RRCTCE Framework)

Before writing any code, say this out loud:

```
R — RESTATE   "So I'm given... and I need to return..."
R — RULES     "Can the array be empty? Are there duplicates? Is it sorted?"
C — CASES     "Edge case: empty input, single element, all same values"
T — THINK     "Brute force is X at O(n²). I can optimize by..."
C — CODE      Write the solution
E — EXAMPLES  "Let me trace through [2,7,11,15] with target 9..."
```

---

## Mock Session 1 — Arrays & Sliding Window

**Problem: Minimum Size Subarray Sum (LeetCode 209)**

> Given an array of positive integers and a positive integer `target`, return the minimal length of a subarray whose sum ≥ target. Return 0 if no such subarray exists.

**Step 1: Restate & Clarify (say out loud)**
```
"So I have an array of positive integers and a target sum.
I need to find the shortest contiguous subarray with sum >= target.

Clarifications:
- All positive? [Yes] — this matters: sum can only grow as window expands
- Can elements be zero? [No per constraint]
- Return 0 if impossible? [Yes]

Edge cases:
- Empty array → 0
- Single element >= target → 1
- All elements sum < target → 0
```

**Step 2: Brute Force**
```
"Naive: check all subarrays — O(n²). For each start i,
expand right until sum >= target, record length.

Can I do better? Yes — sliding window works because
all values are positive. Growing the window increases sum,
shrinking it decreases sum. This lets me use two pointers."
```

**Step 3: Optimal Solution**
```python
def min_subarray_len(target: int, nums: list[int]) -> int:
    left = 0
    current_sum = 0
    min_len = float('inf')

    for right in range(len(nums)):
        current_sum += nums[right]

        # Shrink window while sum is valid
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return min_len if min_len != float('inf') else 0
```

**Step 4: Trace Through Example**
```
nums = [2,3,1,2,4,3], target = 7

right=0: sum=2, no shrink
right=1: sum=5, no shrink
right=2: sum=6, no shrink
right=3: sum=8 >= 7 → record len=4, shrink: sum=6, left=1
right=4: sum=10 >= 7 → record len=4, shrink: sum=7 >= 7 → record len=3, shrink: sum=4, left=3
right=5: sum=7 >= 7 → record len=3, shrink: sum=4, left=4
Answer: 2 (subarray [4,3])

Wait — let me re-check: at right=4, after first shrink sum=7 which is >= target,
so I shrink again: sum=7-2=5 (removed index 3, value 2), left=4.
Then right=5: sum=5+3=8 >= 7 → record len=2 (right=5, left=4) ✓
```

**Step 5: Complexity**
```
Time: O(n) — each element added and removed from window exactly once
Space: O(1)
```

**Interviewer Follow-up Questions:**

**Q: "What if numbers could be negative?"**
> "Sliding window breaks — adding elements to the right no longer guarantees sum increases. I'd need prefix sums + monotonic deque or a different approach. Positive constraint is essential."

**Q: "What if I needed the actual subarray, not just its length?"**
```python
# Track best_left too
best_left = 0
if right - left + 1 < min_len:
    min_len = right - left + 1
    best_left = left
# Return nums[best_left : best_left + min_len]
```

**Q: "Can you solve it in O(n log n) if you don't see the O(n) solution?"**
> "Yes — build a prefix sum array, then for each start index, binary search for the smallest end index where `prefix[end] - prefix[start] >= target`. O(n log n). The O(n) sliding window is better but this shows I know multiple tools."

---

## Mock Session 2 — Trees (Medium)

**Problem: Binary Tree Right Side View (LeetCode 199)**

> Given the root of a binary tree, return the values of nodes you can see if you stand to the right side (the rightmost node at each level).

**Step 1: Restate & Clarify**
```
"Standing to the right, I see the last node at each level.
This is a level-order traversal where I record only the last
element at each level.

Clarifications:
- Can the tree be empty? → return []
- Balanced? → doesn't matter, I need to handle any shape

Edge case: root only → [root.val]
Edge case: only left children → I still see each left child (it's rightmost at its level)
```

**Step 2: Brute Force → Optimal**
```
"BFS naturally gives me nodes level by level. At each level,
I record the last node's value. No optimization needed —
BFS is already O(n)."
```

**Step 3: Solution**
```python
from collections import deque

def right_side_view(root) -> list[int]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)       # snapshot current level

        for i in range(level_size):
            node = queue.popleft()

            if i == level_size - 1:   # last node in this level
                result.append(node.val)

            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)

    return result
```

**Step 4: Trace**
```
Tree:    1
        / \
       2   3
        \   \
         5   4

Level 0: [1]           → see 1
Level 1: [2, 3]        → see 3
Level 2: [5, 4]        → see 4

Output: [1, 3, 4]  ✓
```

**Step 5: DFS Alternative**
```python
def right_side_view_dfs(root) -> list[int]:
    result = []

    def dfs(node, depth):
        if not node:
            return
        if depth == len(result):      # first time reaching this depth
            result.append(node.val)   # will get overwritten if right child exists
        # Visit right FIRST so rightmost node at each depth is seen first
        dfs(node.right, depth + 1)
        dfs(node.left, depth + 1)

    dfs(root, 0)
    return result
```

**Interviewer Follow-up:**

**Q: "What's the difference between your BFS and DFS approaches?"**
> "BFS directly iterates levels, so `len(queue)` gives the level size exactly. DFS works by visiting right children first — the first node we see at each depth is the rightmost one. Both are O(n) time and O(n) space (queue width for BFS, recursion depth for DFS)."

**Q: "What if you needed the left side view?"**
> "BFS: take the first node of each level instead of last. DFS: visit left children first, and take the first node encountered at each depth."

**Q: "What about a tree with 10,000 nodes where every node has only left children (a linked list)?"**
> "DFS would hit Python's recursion limit at ~1,000. I'd use iterative BFS (already written) or convert DFS to use an explicit stack. This is why I prefer BFS for tree level problems."

---

## Mock Session 3 — Dynamic Programming

**Problem: Jump Game (LeetCode 55)**

> Given an array where each element represents your maximum jump length at that position, return `True` if you can reach the last index starting from index 0.

**Step 1: Restate & Clarify**
```
"Each element tells me the max steps I can jump forward.
I start at index 0 and want to reach index n-1.

Clarifications:
- Can elements be 0? Yes — 0 means stuck
- Always at least one element? → handle n=1
- nums[i] >= 0 always? Yes

Edge cases:
- n=1 → already at the end, return True
- nums[0] = 0 and n > 1 → False (can't move)
```

**Step 2: DP Approach (then optimize)**
```
"DP: dp[i] = can I reach index i?
dp[0] = True. For each i, dp[i] = True if any j < i where
dp[j] is True and j + nums[j] >= i.

This is O(n²). Can I do better?

Greedy: instead of tracking which indices are reachable,
just track the furthest index reachable so far.
If at any point my current index > max_reach, I'm stuck."
```

**Step 3: Greedy Solution**
```python
def can_jump(nums: list[int]) -> bool:
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:           # can't reach this index
            return False
        max_reach = max(max_reach, i + nums[i])
    return True

# Example
print(can_jump([2,3,1,1,4]))    # True
print(can_jump([3,2,1,0,4]))    # False (stuck at index 3)
```

**Step 4: Trace**
```
[2, 3, 1, 1, 4]

i=0: max_reach = max(0, 0+2) = 2
i=1: 1 <= 2 ✓, max_reach = max(2, 1+3) = 4
i=2: 2 <= 4 ✓, max_reach = max(4, 2+1) = 4
i=3: 3 <= 4 ✓, max_reach = max(4, 3+1) = 4
i=4: 4 <= 4 ✓, max_reach = max(4, 4+4) = 8
→ True ✓

[3, 2, 1, 0, 4]

i=0: max_reach = 3
i=1: max_reach = max(3, 1+2) = 3
i=2: max_reach = max(3, 2+1) = 3
i=3: max_reach = max(3, 3+0) = 3
i=4: 4 > 3 → return False ✓
```

**Step 5: Complexity**
```
Time: O(n), Space: O(1)
Compare to DP: O(n²), O(n) — greedy is strictly better
```

**Interviewer Follow-up:**

**Q: "Jump Game II — find the minimum number of jumps to reach the end."**
```python
def jump(nums: list[int]) -> int:
    jumps = 0
    current_end = 0    # furthest we can reach with current jumps
    farthest = 0       # furthest we can reach with one more jump

    for i in range(len(nums) - 1):   # don't process last element
        farthest = max(farthest, i + nums[i])
        if i == current_end:          # must make a jump here
            jumps += 1
            current_end = farthest
    return jumps

print(jump([2,3,1,1,4]))    # 2 (0→1→4)
print(jump([2,3,0,1,4]))    # 2 (0→1→4)
```

**Q: "How is this greedy approach provably correct?"**
> "A greedy is correct when a locally optimal choice leads to a globally optimal solution. Here, maintaining `max_reach` is globally correct because: if we can reach index X, we can reach all indices between 0 and X. `max_reach` is a monotone non-decreasing value. At index i, every position between 0 and max_reach is reachable — we never miss anything by taking the maximum."

---

## Scoring Rubric — After Each Session

Rate yourself 1-5 on each:

| Dimension | 1 (Struggle) | 3 (OK) | 5 (Strong) |
|---|---|---|---|
| **Clarification** | Started coding immediately | Asked about edge cases | Identified all constraints before coding |
| **Brute Force** | Couldn't state it | Stated it, no complexity | Stated it, gave complexity, explained why suboptimal |
| **Optimal Approach** | Needed hints | Got there slowly | Explained pattern immediately |
| **Code Quality** | Syntax errors, incomplete | Mostly correct | Clean, handles edges, no bugs |
| **Testing** | Ran without checking | Checked happy path | Traced through, found and fixed a bug |
| **Communication** | Silent | Occasional updates | Continuous narration |
| **Follow-ups** | Blank stare | Partial answers | Clear, structured answers |

**Target for a passing screen:** Average 3+ on all dimensions, 4+ on at least four.

---

## Practice Schedule — 3-Week Sprint

### Week 1: Foundation (Chapter 1 Topics)
```
Day 1: Arrays — Two Sum, Best Time to Buy/Sell, Contains Duplicate
Day 2: Strings — Valid Anagram, Longest Common Prefix
Day 3: Hash Maps — Group Anagrams, Top K Frequent
Day 4: Trees — Max Depth, Invert Tree, Symmetric Tree
Day 5: Recursion — Climbing Stairs, Generate Parentheses
Day 6-7: Review + redo any problems you struggled with
```

### Week 2: Patterns (Chapter 2 Topics)
```
Day 1: Sliding Window — Longest Substring No Repeat, Min Window Substring
Day 2: Two Pointers — 3Sum, Container With Most Water
Day 3: BFS/DFS — Course Schedule, Rotting Oranges
Day 4: Backtracking — Subsets, Permutations, Combination Sum
Day 5: Binary Search — Rotated Array, Koko Bananas
Day 6: DP — Coin Change II, LCS, Partition Equal Subset
Day 7: Mock session (pick any 1 problem per pattern)
```

### Week 3: Mock Interviews (Chapter 3 Format)
```
Day 1-2: Full 45-min mock per day (1 problem)
Day 3-4: Full 45-min mock, then review the solution in depth
Day 5: Revisit your 3 weakest topics
Day 6: Mock with a friend or record yourself
Day 7: Rest — review complexity table, pattern cheat sheet
```

---

## Quick Reference: Complexity Cheat Sheet

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| Array | O(1) | O(n) | O(n) | O(n) |
| Sorted Array | O(1) | O(log n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) at head | O(1) with ptr |
| Hash Map | O(1) avg | O(1) avg | O(1) avg | O(1) avg |
| Hash Set | — | O(1) avg | O(1) avg | O(1) avg |
| Binary Search Tree | O(log n) avg | O(log n) avg | O(log n) avg | O(log n) avg |
| Heap (min/max) | O(1) peek | O(n) | O(log n) | O(log n) |
| Deque | O(1) ends | O(n) | O(1) ends | O(1) ends |

| Algorithm | Best | Average | Worst | Space |
|---|---|---|---|---|
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| BFS/DFS | — | O(V+E) | O(V+E) | O(V) |
| Quicksort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Mergesort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Heapsort | O(n log n) | O(n log n) | O(n log n) | O(1) |

---

## Blind 75 — Topic Mapping

Use this as your practice list. Chapters 1 and 2 cover all patterns needed.

**Arrays (sorted into patterns):**
- Sliding Window: 3, 76, 121, 239, 567
- Two Pointers: 11, 15, 42, 125
- General: 1, 217, 238, 53, 152, 153, 33

**Binary Search:** 153, 33, 74, 278

**Hash Map/Set:** 1, 49, 128, 347, 242, 383

**Trees:** 104, 226, 101, 543, 124, 102, 297, 572, 105, 98, 230, 235, 208

**Graphs:** 200, 133, 207, 417, 994, 323

**Backtracking:** 79, 39, 46, 78, 51

**DP:** 70, 198, 213, 91, 300, 1143, 139, 322, 377, 416, 62, 64

---

## Final Tips for the Day of the Interview

1. **Clarify before coding** — 2 minutes of questions prevents 20 minutes of wrong solution
2. **Say the brute force** — even if you know the optimal, stating brute force shows structured thinking
3. **Name the pattern** — "This looks like a sliding window problem" signals pattern recognition
4. **Code clean, not clever** — readable variable names, no one-liners that require explanation
5. **Test with a simple example** — pick a 4-5 element input and trace through manually
6. **State complexity unprompted** — before the interviewer asks: "This is O(n) time and O(1) space"
7. **If stuck, think out loud** — "I know I need to avoid redundant work... that suggests caching or a smarter traversal..." Silence is the only wrong answer
