# Chapter 1 — Section 2: Hash Maps & Sets

## Core Concepts

A **hash map** (dict in Python) maps keys → values in O(1) average for get, set, and delete.
A **hash set** gives O(1) membership testing.

**When to reach for them:**
- You need to count occurrences of something (frequency map)
- You need O(1) lookup by value (complement lookup, deduplication)
- You need to group things by a common property

---

## Q&A 1 — Group Anagrams

**Q:** Given a list of strings, group the anagrams together.

**What the interviewer is watching for:** Can you identify a canonical "key" that all anagrams share?

**A — Think out loud:**
> "Two strings are anagrams if they have the same characters in the same frequencies. The sorted version is the same for all anagrams in a group — I can use that as a hash map key."

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))     # "eat" → ('a','e','t')
        groups[key].append(s)
    return list(groups.values())

# Example
print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# [['eat','tea','ate'], ['tan','nat'], ['bat']]
```

**Complexity:** Time O(n · k log k) where k is max string length, Space O(nk)

**Optimized key** (avoids sort, O(k) instead of O(k log k)):
```python
def group_anagrams_fast(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)
    return list(groups.values())
```

---

## Q&A 2 — Top K Frequent Elements

**Q:** Given an integer array, return the `k` most frequent elements.

**A — Build frequency map, then find top K:**

```python
import heapq
from collections import Counter

# Approach 1: Sort by frequency — O(n log n)
def top_k_frequent_sort(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    return [num for num, _ in count.most_common(k)]

# Approach 2: Min-heap — O(n log k), better when k << n
def top_k_frequent_heap(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Approach 3: Bucket sort — O(n), best
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    # buckets[i] = list of numbers that appear exactly i times
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    result = []
    for freq in range(len(buckets) - 1, 0, -1):   # high freq first
        result.extend(buckets[freq])
        if len(result) >= k:
            return result[:k]
    return result

# Example
print(top_k_frequent([1,1,1,2,2,3], 2))  # [1, 2]
```

**Say the trade-off clearly:** "Sort is O(n log n) and simplest to explain. Heap is O(n log k). Bucket sort is O(n) but takes extra space — great if the interviewer asks to optimize."

---

## Q&A 3 — Valid Sudoku

**Q:** Determine if a 9×9 Sudoku board is valid (each row, column, and 3×3 box has no repeats among 1–9).

**A — Use sets to track what you've seen:**
```python
def is_valid_sudoku(board: list[list[str]]) -> bool:
    rows    = [set() for _ in range(9)]
    cols    = [set() for _ in range(9)]
    boxes   = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue

            box_idx = (r // 3) * 3 + (c // 3)   # 0..8

            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False

            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)

    return True
```

**Complexity:** Time O(81) = O(1), Space O(81) = O(1) — fixed-size board.

**Key insight to highlight:** "The box index formula `(r // 3) * 3 + (c // 3)` maps any cell to one of the nine 3×3 boxes."

---

## Q&A 4 — Longest Consecutive Sequence

**Q:** Given an unsorted array, find the length of the longest consecutive sequence (e.g., [100,4,200,1,3,2] → 4 because 1,2,3,4).

**A — The trick is figuring out where a sequence starts:**
```python
def longest_consecutive(nums: list[int]) -> int:
    num_set = set(nums)
    best = 0

    for n in num_set:
        # Only start counting from the beginning of a sequence
        if n - 1 not in num_set:
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)

    return best

# Example
print(longest_consecutive([100,4,200,1,3,2]))  # 4
```

**Complexity:** Time O(n) amortized (each number visited at most twice), Space O(n)

**Why O(n) not O(n²):** "The inner while loop only runs when we're at the start of a sequence. Across the whole array, each element is visited by the while loop at most once total."

---

## Q&A 5 — Ransom Note

**Q:** Given two strings `ransomNote` and `magazine`, return `True` if you can construct `ransomNote` using letters from `magazine` (each letter can only be used once).

```python
from collections import Counter

def can_construct(ransom_note: str, magazine: str) -> bool:
    mag_count = Counter(magazine)
    for c in ransom_note:
        if mag_count[c] <= 0:
            return False
        mag_count[c] -= 1
    return True

# Cleaner one-liner
def can_construct(ransom_note: str, magazine: str) -> bool:
    return not (Counter(ransom_note) - Counter(magazine))
```

**Counter subtraction semantics:** `Counter("aab") - Counter("ab")` = `Counter({'a': 1})`. If the result is empty (falsy), ransom note can be built.

---

## Q&A 6 — Interview Dialogue: Hash Map vs Array as Counter

**Interviewer:** "You used a Counter. Could you use a plain array instead?"

**You:** "Yes — if I know the character set is fixed (e.g., lowercase ASCII), I can use `count = [0] * 26` and index with `ord(c) - ord('a')`. That's slightly faster in practice and uses O(1) space relative to the alphabet size rather than O(unique chars). For unicode or unknown character sets, Counter is safer."

```python
def is_anagram_array(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count = [0] * 26
    for c in s:
        count[ord(c) - ord('a')] += 1
    for c in t:
        count[ord(c) - ord('a')] -= 1
        if count[ord(c) - ord('a')] < 0:
            return False
    return True
```

---

## Pattern Recognition Cheat Sheet

| Signal | Hash Map / Set pattern |
|---|---|
| "find pair that sums to X" | Complement map: `seen[target - num]` |
| "group by shared property" | `defaultdict(list)`, canonical key |
| "count occurrences" | `Counter` or `defaultdict(int)` |
| "check membership fast" | `set()` |
| "first unique / duplicate" | `OrderedDict` or insertion-order dict |
| "detect cycle" | `set` of visited nodes |

---

## Common Mistakes

1. **Using a list for O(1) lookup** — lists are O(n) for `in`; sets are O(1)
2. **Forgetting `defaultdict`** — avoid `KeyError` with `defaultdict(int)` or `.get(key, 0)`
3. **Mutating while iterating** — iterate over a copy if you modify the map mid-loop
4. **Hash collisions in worst case** — technically O(n) worst case, but always say "O(1) average" in interviews
