# Coding Interview Q&A Book

A practical, pattern-focused guide to passing technical screening interviews.
Built around the **NeetCode Blind 75** pattern set — the same problems top companies use.

---

## Who This Is For

Engineers preparing for a screening-stage technical interview (45–60 min, 1–2 problems).
The book assumes basic programming knowledge and focuses on recognizing patterns fast.

---

## What the Screener Tests

| Topic | Depth Needed | Key Patterns |
|---|---|---|
| Arrays & Strings | Medium | Sliding window, two pointers |
| Hash Maps / Sets | Medium | Frequency counts, O(1) lookup |
| Trees & Graphs | Medium | BFS, DFS, basic traversal |
| Recursion & Backtracking | Medium | Decision trees, undo |
| Sorting & Binary Search | Medium | Know when to apply |
| Dynamic Programming | Recognize | 1D DP patterns only |
| Advanced Graph Algorithms | **Not tested** | Skip at screening stage |
| Segment Trees, AVL, etc. | **Not tested** | Skip at screening stage |

---

## Book Structure

### Chapter 1 — Core Data Structures (Easy Problems)
Build understanding through easy LeetCode problems, one topic at a time.

| Section | Topic | Problems Covered |
|---|---|---|
| [1.1](chapter_1/01_arrays_strings.md) | Arrays & Strings | Two Sum, Best Time to Buy/Sell, Contains Duplicate, Valid Anagram, Reverse String |
| [1.2](chapter_1/02_hashmaps_sets.md) | Hash Maps & Sets | Group Anagrams, Top K Frequent, Valid Sudoku, Longest Consecutive, Ransom Note |
| [1.3](chapter_1/03_trees_graphs.md) | Trees & Graphs | Max Depth, Invert Tree, Symmetric Tree, Path Sum, Number of Islands, Level Order |
| [1.4](chapter_1/04_recursion_backtracking.md) | Recursion & Backtracking | Fibonacci, Power, Generate Parentheses, Climbing Stairs, Letter Combinations |
| [1.5](chapter_1/05_sorting_binary_search.md) | Sorting & Binary Search | Binary Search, Search Insert, First Bad Version, Find Min Rotated, Kth Largest |
| [1.6](chapter_1/06_dynamic_programming.md) | Dynamic Programming | Climbing Stairs, House Robber, Max Subarray, Coin Change, LIS, Decode Ways |

### Chapter 2 — Pattern Mastery (Medium Problems)
Focus on patterns, not memorization. Each section teaches a transferable technique.

| Section | Pattern | Problems Covered |
|---|---|---|
| [2.1](chapter_2/01_sliding_window.md) | Sliding Window | Longest Substring No Repeat, Max Sum Subarray K, Min Window Substring, Char Replacement, Permutation in String |
| [2.2](chapter_2/02_two_pointers.md) | Two Pointers | Two Sum II, 3Sum, Container With Most Water, Remove Duplicates, Trapping Rain Water, Valid Palindrome |
| [2.3](chapter_2/03_bfs_dfs_medium.md) | BFS & DFS | Clone Graph, Course Schedule, Word Ladder, Pacific Atlantic, Rotting Oranges |
| [2.4](chapter_2/04_backtracking_medium.md) | Backtracking | Subsets, Permutations, Combination Sum, Word Search, N-Queens |
| [2.5](chapter_2/05_binary_search_medium.md) | Binary Search | Rotated Array, Find Peak, Koko Bananas, Ship Packages, Time-Based K-V Store |
| [2.6](chapter_2/06_dynamic_programming_medium.md) | Dynamic Programming | Unique Paths, Coin Change II, LCS, Stock Cooldown, Partition Subset Sum, Decode Ways |

### Chapter 3 — Mock Interview Sessions
45-minute timed sessions with full transcripts, follow-up Q&As, and a scoring rubric.

| Session | Topics | Problem |
|---|---|---|
| [Session 1](chapter_3/mock_sessions.md#mock-session-1--arrays--sliding-window) | Arrays, Sliding Window | Min Size Subarray Sum |
| [Session 2](chapter_3/mock_sessions.md#mock-session-2--trees-medium) | Trees, BFS | Binary Tree Right Side View |
| [Session 3](chapter_3/mock_sessions.md#mock-session-3--dynamic-programming) | Greedy, DP | Jump Game |

---

## How to Study

### Week 1: Chapter 1 — Build the foundation
Work through each section. For every problem:
1. Read the Q&A
2. Close the book and code it from memory
3. Check your solution, understand the differences

### Week 2: Chapter 2 — Learn the patterns
For each pattern section:
1. Understand the template
2. Code the problems in the section
3. Find 2–3 more problems of the same pattern on LeetCode

### Week 3: Chapter 3 — Simulate interviews
- Set a 45-minute timer per session
- Talk out loud — narrate your entire thought process
- Grade yourself with the rubric after each session
- Find a friend or use a recording to review your communication

---

## Pattern Recognition — Quick Reference

When you see this in the problem → reach for this pattern:

```
"Longest/shortest subarray with condition"    → Sliding window
"Find pair that sums to X in sorted array"    → Two pointers
"Find pair that sums to X in unsorted array"  → Hash map complement
"Shortest path / minimum steps"               → BFS
"All paths / generate all combinations"       → DFS + backtracking
"Binary tree level by level"                  → BFS with len(queue) snapshot
"Sorted array, find value"                    → Binary search
"Minimum X such that condition holds"         → Binary search on answer space
"Count/maximize ways to reach a total"        → DP (knapsack)
"Two sequences, find longest common..."       → 2D DP
"Each choice has two options at each step"    → State machine DP or backtracking
"Group by shared property"                    → Hash map with canonical key
"Cycle detection in directed graph"           → DFS with 3-color marking
"Topological order"                           → Kahn's BFS (in-degree)
```

---

## Complexity Reference

**Target complexities for a passing answer:**
- Easy: O(n) time, O(1)–O(n) space
- Medium: O(n log n) or O(n) time, O(n) space
- Hard: O(n²) is usually acceptable; O(n log n) gets bonus points

**Always state complexity before the interviewer asks.**

---

## Resources

- **NeetCode.io** — video walkthroughs organized by pattern (highly recommended)
- **LeetCode Blind 75** — the canonical targeted list (mapped in Chapter 3)
- **Python `collections` module** — `Counter`, `defaultdict`, `deque` are essential
- **`heapq` module** — for priority queues (min-heap by default)
- **`bisect` module** — for binary search on sorted lists

---

## Common Python Tricks

```python
# Frequency count
from collections import Counter
count = Counter("abracadabra")   # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# Default dict (no KeyError)
from collections import defaultdict
graph = defaultdict(list)

# BFS queue (O(1) popleft)
from collections import deque
q = deque([start])
q.popleft()   # NOT q.pop(0) which is O(n)

# Min-heap
import heapq
heap = []
heapq.heappush(heap, val)
smallest = heapq.heappop(heap)

# Max-heap (negate values)
heapq.heappush(heap, -val)
largest = -heapq.heappop(heap)

# Binary search
from bisect import bisect_left, bisect_right
i = bisect_left(sorted_list, target)   # leftmost position for target

# Sort by custom key
intervals.sort(key=lambda x: x[0])    # sort by start time

# Infinity
float('inf'), float('-inf')

# Integer division (floor)
mid = lo + (hi - lo) // 2

# String to list and back
chars = list(s)
result = ''.join(chars)
```
