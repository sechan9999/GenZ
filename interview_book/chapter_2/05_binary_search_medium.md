# Chapter 2 — Pattern 5: Binary Search (Medium)

## The Core Insight

Binary search works whenever you can answer the question: **"Given a candidate answer X, can I verify in O(n) whether X is valid?"** and the set of valid/invalid answers is monotone (all invalid come before all valid, or vice versa).

**Two templates to master:**

```python
# Template A: Find exact value (closed interval)
lo, hi = 0, len(nums) - 1
while lo <= hi:
    mid = lo + (hi - lo) // 2
    if check(mid) == target:   return mid
    elif check(mid) < target:  lo = mid + 1
    else:                      hi = mid - 1
return -1

# Template B: Find first True (open right boundary)
lo, hi = lower_bound, upper_bound
while lo < hi:
    mid = (lo + hi) // 2
    if condition(mid):   hi = mid       # mid could be the answer
    else:                lo = mid + 1   # mid is definitely not the answer
return lo   # lo == hi, converged
```

---

## Q&A 1 — Search in Rotated Sorted Array

**Q:** A sorted array was rotated at some pivot. Search for a target value.

**A — Modified binary search on which half is sorted:**

```python
def search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1             # target in sorted left half
            else:
                lo = mid + 1             # target must be in right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1             # target in sorted right half
            else:
                hi = mid - 1             # target must be in left half

    return -1

# Example
print(search([4,5,6,7,0,1,2], 0))    # 4
print(search([4,5,6,7,0,1,2], 3))    # -1
print(search([1], 0))                 # -1
```

**Complexity:** Time O(log n), Space O(1)

**Explain the logic:** "In a rotated array, at least one half is always sorted. I check which half is sorted by comparing `nums[lo]` to `nums[mid]`. Then I check if the target falls inside the sorted half — if yes, binary search there; if no, search the other half."

---

## Q&A 2 — Find Peak Element

**Q:** A peak element is greater than its neighbors. Find the index of any peak. Assume `nums[-1] = nums[n] = -∞`.

**A — Binary search: always move toward the rising slope:**

```python
def find_peak_element(nums: list[int]) -> int:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[mid + 1]:
            hi = mid           # peak is at mid or to the left
        else:
            lo = mid + 1       # peak is to the right of mid
    return lo

# Example
print(find_peak_element([1,2,3,1]))       # 2 (index of value 3)
print(find_peak_element([1,2,1,3,5,6,4])) # 5 or 1 (any peak is ok)
```

**Complexity:** Time O(log n), Space O(1)

**Explain why O(log n) works on unsorted data:** "Even though the array isn't sorted, the slope is monotone locally. If `nums[mid] < nums[mid+1]`, then a peak must exist to the right — the values are rising, and since they must eventually stop (bounded by -∞), there's a peak somewhere in that direction."

---

## Q&A 3 — Koko Eating Bananas (Binary Search on Answer)

**Q:** Koko can eat at most k bananas per hour from a pile. Given n piles and h hours, find the minimum k such that she finishes in h hours.

**A — Binary search on k (eating speed):**

```python
import math

def min_eating_speed(piles: list[int], h: int) -> int:
    def can_finish(speed):
        hours = sum(math.ceil(pile / speed) for pile in piles)
        return hours <= h

    lo, hi = 1, max(piles)           # min speed 1, max speed = largest pile
    while lo < hi:
        mid = (lo + hi) // 2
        if can_finish(mid):
            hi = mid                  # mid works, maybe smaller does too
        else:
            lo = mid + 1             # mid too slow, need faster

    return lo

# Example
print(min_eating_speed([3,6,7,11], 8))   # 4
print(min_eating_speed([30,11,23,4,20], 5))  # 30
```

**Complexity:** Time O(n log(max_pile)), Space O(1)

**The "binary search on answer" template:**
> "I binary search on the answer space `[1, max(piles)]`. For each candidate speed `mid`, I check if it's feasible in O(n). The predicate is monotone: if speed `k` works, so does `k+1`. I find the minimum that works."

---

## Q&A 4 — Capacity to Ship Packages Within D Days

**Q:** A conveyor belt has packages with given weights. The ship carries a fixed capacity per day and must ship all packages in order within D days. Find the minimum capacity.

**A — Same "binary search on answer" pattern:**

```python
def ship_within_days(weights: list[int], days: int) -> int:
    def can_ship(capacity):
        day_count, current = 1, 0
        for w in weights:
            if current + w > capacity:
                day_count += 1
                current = 0
            current += w
        return day_count <= days

    lo = max(weights)          # min capacity: must carry heaviest package
    hi = sum(weights)          # max capacity: ship everything in one day

    while lo < hi:
        mid = (lo + hi) // 2
        if can_ship(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo

# Example
print(ship_within_days([1,2,3,4,5,6,7,8,9,10], 5))    # 15
print(ship_within_days([3,2,2,4,1,4], 3))               # 6
```

**Complexity:** Time O(n log(sum-max)), Space O(1)

---

## Q&A 5 — Time-Based Key-Value Store

**Q:** Design a data structure that supports `set(key, value, timestamp)` and `get(key, timestamp)` — `get` returns the most recent value set at or before the given timestamp.

**A — Store sorted timestamps, binary search on get:**

```python
from collections import defaultdict
from bisect import bisect_right

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)   # key → [(timestamp, value)]

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))   # timestamps are always increasing

    def get(self, key: str, timestamp: int) -> str:
        entries = self.store[key]
        if not entries:
            return ""
        # Find rightmost timestamp <= given timestamp
        lo, hi = 0, len(entries) - 1
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            if entries[mid][0] <= timestamp:
                result = entries[mid][1]   # valid candidate, try to go right
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # Cleaner with bisect:
    def get_bisect(self, key: str, timestamp: int) -> str:
        entries = self.store[key]
        # bisect_right returns insertion point after all timestamps <= given
        i = bisect_right(entries, (timestamp, chr(127)))   # chr(127) > any char
        return entries[i-1][1] if i > 0 else ""
```

**Complexity:** `set` O(1), `get` O(log n)

---

## Q&A 6 — Interview Dialogue: Recognizing "Binary Search on Answer"

**Interviewer:** "How do you recognize when to binary search on the answer space vs the array directly?"

**You:** "I ask: 'Is there a monotone relationship between the answer and feasibility?' For example:
- 'Minimum capacity to ship in D days' — if capacity C works, then C+1 also works. That's monotone → binary search on capacity.
- 'Minimum speed to eat all bananas in H hours' — if speed k works, k+1 also works. Monotone → binary search on speed.
- 'Minimum days to divide books with max pages K' — if K pages works for M students, K+1 also works.

The signal is: the problem asks for a minimum/maximum value, there's a verification function `feasible(candidate)` that runs in O(n), and `feasible` is a step function (all False then all True, or vice versa).

I set `lo` and `hi` to the natural bounds of the answer, binary search, and call `feasible(mid)` at each step."

---

## Pattern Recognition Cheat Sheet

| Problem type | Binary search target |
|---|---|
| Find value in sorted array | Array index |
| Find first True in monotone predicate | Predicate space |
| Minimum feasible value (monotone) | Answer space |
| Rotated sorted array | Modified: determine which half is sorted |
| Peak finding | Modified: follow rising slope |
| `k`-th smallest | Answer space (binary search on value, not index) |

---

## Common Mistakes

1. **Not establishing the monotone property** — always verify that if X works, X+1 also works (or vice versa)
2. **Wrong lo/hi bounds** — `lo = max(weights)` for shipping (can't carry less than heaviest), not `lo = 1`
3. **Mixing up `< ` and `<=` in the loop** — Template B uses `while lo < hi`; be consistent with boundary updates
4. **Not handling equal timestamps in TimeMap** — `bisect_right` on `(timestamp, big_char)` handles this cleanly
