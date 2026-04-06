# Chapter 2 — Pattern 1: Sliding Window

## The Pattern

A **sliding window** is a subarray (or substring) of variable or fixed size that you move across the data without restarting from scratch. It converts an O(n²) nested loop into O(n) by maintaining a running state.

**Two variants:**
- **Fixed size:** window size k stays constant — just add the new right element and remove the old left element
- **Variable size:** window expands to the right; shrinks from the left when a condition is violated

**The template:**
```python
def sliding_window(nums, k):
    left = 0
    window_state = ...          # running sum, char count, max, etc.
    result = ...

    for right in range(len(nums)):
        # 1. Expand: add nums[right] to window
        window_state = update(window_state, nums[right])

        # 2. Shrink: remove from left while window is invalid
        while window_is_invalid(window_state):
            window_state = remove(window_state, nums[left])
            left += 1

        # 3. Record result at this window position
        result = best(result, right - left + 1)

    return result
```

---

## Q&A 1 — Longest Substring Without Repeating Characters

**Q:** Find the length of the longest substring without repeating characters.

**A — Variable window, shrink when duplicate appears:**

```python
def length_of_longest_substring(s: str) -> int:
    char_index = {}   # char → last seen index
    left = 0
    best = 0

    for right, c in enumerate(s):
        # If c was seen and its last position is inside current window
        if c in char_index and char_index[c] >= left:
            left = char_index[c] + 1   # jump left past the duplicate

        char_index[c] = right
        best = max(best, right - left + 1)

    return best

# Examples
print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
print(length_of_longest_substring("bbbbb"))     # 1 ("b")
print(length_of_longest_substring("pwwkew"))    # 3 ("wke")
```

**Complexity:** Time O(n), Space O(min(n, alphabet))

**Explain the jump:** "Instead of moving left one step at a time, I jump directly to `last_seen[c] + 1`. This skips past all invalid positions in one move. I check `char_index[c] >= left` to ignore occurrences outside the current window."

---

## Q&A 2 — Maximum Sum Subarray of Size K (Fixed Window)

**Q:** Find the maximum sum of any contiguous subarray of size k.

**A — Fixed window, slide one step at a time:**

```python
def max_sum_subarray(nums: list[int], k: int) -> int:
    if len(nums) < k:
        return 0

    # Build the first window
    window_sum = sum(nums[:k])
    best = window_sum

    # Slide: add right element, remove left element
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)

    return best

# Example
print(max_sum_subarray([2,1,5,1,3,2], 3))  # 9  (5+1+3)
print(max_sum_subarray([2,3,4,1,5], 2))    # 7  (3+4)
```

**Complexity:** Time O(n), Space O(1)

---

## Q&A 3 — Minimum Window Substring

**Q:** Given strings `s` and `t`, return the minimum window substring of `s` that contains all characters of `t`. (Hard)

**A — Variable window, track a "satisfied" counter:**

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""

    need = Counter(t)          # how many of each char we still need
    missing = len(t)           # total chars still missing from window
    left = 0
    result_start, result_len = 0, float('inf')

    for right, c in enumerate(s):
        # Expand: character c helps if we still need it
        if need[c] > 0:
            missing -= 1
        need[c] -= 1           # track excess as negative

        # Shrink: once window satisfies t, try to shrink from left
        while missing == 0:
            if right - left + 1 < result_len:
                result_start = left
                result_len = right - left + 1

            # Remove left char from window
            left_c = s[left]
            need[left_c] += 1
            if need[left_c] > 0:   # we lost a necessary character
                missing += 1
            left += 1

    return s[result_start:result_start + result_len] if result_len != float('inf') else ""

# Example
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
print(min_window("a", "a"))               # "a"
print(min_window("a", "aa"))              # ""
```

**Complexity:** Time O(|s| + |t|), Space O(|s| + |t|)

**Explain the `missing` counter:**
> "I track how many characters are still missing from the window. Adding a character decrements `missing` only if we still needed more of it (`need[c] > 0`). Negative `need` counts mean we have excess — those don't count as satisfied. When `missing == 0`, the window is valid and I try to shrink it."

---

## Q&A 4 — Longest Repeating Character Replacement

**Q:** Given a string and an integer k, find the length of the longest substring containing the same letter after replacing at most k characters.

**A — Key insight: track the most frequent char in the window:**

```python
def character_replacement(s: str, k: int) -> int:
    count = {}
    left = 0
    max_count = 0    # frequency of the most common char in window
    result = 0

    for right, c in enumerate(s):
        count[c] = count.get(c, 0) + 1
        max_count = max(max_count, count[c])

        # Window size - max_count = chars we need to replace
        window_size = right - left + 1
        if window_size - max_count > k:
            count[s[left]] -= 1
            left += 1

        result = max(result, right - left + 1)

    return result

# Example
print(character_replacement("ABAB", 2))   # 4 (replace both 'A' or both 'B')
print(character_replacement("AABABBA", 1)) # 4
```

**Complexity:** Time O(n), Space O(1) (26 chars max)

**The clever trick:** "Window validity = `window_size - max_count <= k`. If the window has 10 chars and the most common appears 7 times, we only need 3 replacements. I never need to decrease `max_count` — even if it becomes stale, the result only increases when we find a strictly longer valid window."

---

## Q&A 5 — Interview Dialogue: Choosing Fixed vs Variable Window

**Interviewer:** "How do you decide between a fixed and variable window?"

**You:** "If the problem says 'subarray of size k' or gives a specific window size, it's fixed. Fixed windows just add right and remove left in lockstep — very straightforward. If the problem says 'longest/shortest subarray satisfying condition X', it's variable. I expand right always, and shrink left when the condition is violated. The key is identifying what condition makes the window valid or invalid."

**Common condition types:**
| Condition | Window is invalid when... |
|---|---|
| At most k distinct chars | `len(char_count) > k` |
| No repeating chars | `count[c] > 1` |
| Sum ≤ target | `window_sum > target` |
| k replacements | `window_size - max_count > k` |
| Contains all of t | `missing > 0` (shrink while valid) |

---

## Q&A 6 — Permutation in String

**Q:** Given strings `s1` and `s2`, return `True` if any permutation of `s1` is a substring of `s2`.

**A — Fixed window of size `len(s1)`, compare frequency maps:**

```python
from collections import Counter

def check_inclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False

    need = Counter(s1)
    window = Counter(s2[:len(s1)])    # first window

    if window == need:
        return True

    for i in range(len(s1), len(s2)):
        # Add new right char
        new_c = s2[i]
        window[new_c] += 1

        # Remove old left char
        old_c = s2[i - len(s1)]
        window[old_c] -= 1
        if window[old_c] == 0:
            del window[old_c]         # keep Counter clean

        if window == need:
            return True

    return False

# Example
print(check_inclusion("ab", "eidbaooo"))  # True  ("ba" at index 3)
print(check_inclusion("ab", "eidboaoo"))  # False
```

**Complexity:** Time O(|s1| + |s2|), Space O(1) (26 chars)

---

## Pattern Recognition Cheat Sheet

| Problem shape | Window type | Key state |
|---|---|---|
| Subarray sum = target | Variable | running sum |
| Longest with k distinct | Variable | `len(char_count)` or distinct counter |
| Fixed size subarray max/sum | Fixed | running sum |
| Contains all of t | Variable | `missing` count |
| Permutation as substring | Fixed | frequency Counter |
| Longest without repeats | Variable | last seen index |

---

## Common Mistakes

1. **Using `while` instead of `if` for shrinking** — in some patterns you shrink exactly once (e.g., character replacement); in others you shrink until valid (e.g., min window)
2. **Not copying state before recording result** — window boundaries change; record the right-left+1 length, not the window itself
3. **Forgetting to check result after the loop** — some patterns record the result at each step; others only record when condition is met
4. **Starting left past 0** — always initialize `left = 0`
