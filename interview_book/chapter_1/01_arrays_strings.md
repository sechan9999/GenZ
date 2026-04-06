# Chapter 1 — Section 1: Arrays & Strings

## Core Concepts

Arrays give O(1) index access but O(n) insert/delete (except at the end). Strings are usually immutable — concatenation in a loop is O(n²); use a list and `join()` instead.

**The two key patterns for arrays/strings:**
- **Two pointers** — one from each end, or one slow/one fast
- **Sliding window** — a subarray that expands/shrinks as you move right

---

## Q&A 1 — Two Sum

**Q:** Given an integer array `nums` and a target, return indices of the two numbers that add up to the target.

**What the interviewer is looking for:**
Can you move from the brute-force O(n²) answer to O(n) using a hash map?

**A — Think out loud like this:**
> "Brute force is nested loops — check every pair. That's O(n²). If I store each number I've seen in a hash map with its index, then for each new number I just check whether `target - num` is already in the map. One pass, O(n)."

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}                          # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Example
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
print(two_sum([3, 2, 4], 6))       # [1, 2]
```

**Complexity:** Time O(n), Space O(n)

**Follow-up the interviewer might ask:**
- "What if the array is sorted?" → Use two pointers, O(1) space
- "What if there could be multiple pairs?" → Collect all, still O(n)

---

## Q&A 2 — Best Time to Buy and Sell Stock

**Q:** Given prices where `prices[i]` is the price on day `i`, find the maximum profit from one buy and one sell (must buy before you sell).

**A — Think out loud:**
> "I need the biggest difference where the smaller value comes first. Instead of checking all pairs O(n²), I track the minimum price seen so far and at each step check if selling today beats my best profit."

```python
def max_profit(prices: list[int]) -> int:
    min_price = float('inf')
    best = 0
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > best:
            best = price - min_price
    return best

# Example
print(max_profit([7, 1, 5, 3, 6, 4]))  # 5  (buy at 1, sell at 6)
print(max_profit([7, 6, 4, 3, 1]))     # 0  (prices always fall)
```

**Complexity:** Time O(n), Space O(1)

**Key insight to say aloud:** "I never need to look back — the minimum before today is enough."

---

## Q&A 3 — Contains Duplicate

**Q:** Return `True` if any value appears at least twice.

**A:**
```python
def contains_duplicate(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))

# Or equivalently with early exit:
def contains_duplicate(nums: list[int]) -> bool:
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False
```

**Complexity:** Time O(n), Space O(n)

**Don't overthink easy problems.** Say: "A set has O(1) lookup. If any element is already in the set when I try to insert, it's a duplicate."

---

## Q&A 4 — Valid Anagram

**Q:** Return `True` if `t` is an anagram of `s` (same characters, same counts).

**A — Two approaches:**

```python
# Approach 1: Sort both — simple, O(n log n)
def is_anagram_sort(s: str, t: str) -> bool:
    return sorted(s) == sorted(t)

# Approach 2: Frequency count — O(n), preferred
def is_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1
    for c in t:
        if c not in count or count[c] == 0:
            return False
        count[c] -= 1
    return True

# Python shortcut
from collections import Counter
def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)
```

**When to prefer sorting:** When you want simpler code and n is small.
**When to prefer counting:** When you need O(n) time strictly.

---

## Q&A 5 — Reverse a String

**Q:** Reverse string `s` in-place (given as a list of characters).

**A:**
```python
def reverse_string(s: list[str]) -> None:
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
```

**This is the canonical two-pointer swap.** Memorize this pattern — it appears everywhere (palindrome check, partition, etc.).

---

## Q&A 6 — Longest Common Prefix

**Q:** Find the longest common prefix among an array of strings.

**A — Think column-by-column:**
```python
def longest_common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        # Shrink prefix until s starts with it
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

# Example
print(longest_common_prefix(["flower","flow","flight"]))  # "fl"
print(longest_common_prefix(["dog","racecar","car"]))     # ""
```

**Complexity:** O(S) where S = total characters across all strings.

---

## Pattern Recognition Cheat Sheet

| Signal in the problem | Pattern to reach for |
|---|---|
| "subarray/substring with condition" | Sliding window |
| "two elements that satisfy X" | Two pointers |
| "find pair that sums to target" | Hash map complement lookup |
| "does arrangement matter?" | Sort first |
| "check characters/frequency" | Counter / frequency array |

---

## Common Mistakes

1. **Off-by-one in two pointers** — draw the array, trace your loop condition
2. **Mutating a string** — in Python, strings are immutable; build a list then join
3. **Forgetting empty input** — always ask "what if the array is empty?"
4. **Using sort when O(n) is possible** — frequency map beats sort for anagram problems
