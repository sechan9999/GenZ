# Chapter 1 — Section 5: Sorting & Binary Search

## Core Concepts

**Sorting complexity you must know:**
| Algorithm | Time Best | Time Avg | Time Worst | Space | Stable? |
|---|---|---|---|---|---|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |
| Python's `sort()` | — | O(n log n) | O(n log n) | O(n) | Yes (Timsort) |

**Binary search requires:** sorted array (or a sorted-like property — monotone function)

**The binary search invariant:** at every step, the answer is within `[lo, hi]`. Your loop condition and boundary updates must preserve this.

---

## Q&A 1 — Binary Search (Classic)

**Q:** Given a sorted array and a target, return its index or -1 if not present.

**A — The canonical template (closed interval [lo, hi]):**
```python
def binary_search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2   # avoids integer overflow (important in Java/C++)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# Example
print(binary_search([-1,0,3,5,9,12], 9))   # 4
print(binary_search([-1,0,3,5,9,12], 2))   # -1
```

**Complexity:** Time O(log n), Space O(1)

**Explain the off-by-one:** "I use `lo <= hi` so we check when the search space is exactly one element. `mid = lo + (hi - lo) // 2` avoids overflow — in Python integers are unbounded, but it's good practice."

---

## Q&A 2 — Search Insert Position

**Q:** Given a sorted array and a target, return the index where it is, or where it would be inserted.

**A — This is "find the leftmost position where `nums[pos] >= target`":**
```python
def search_insert(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums)             # hi = n (target may go at end)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Example
print(search_insert([1,3,5,6], 5))   # 2
print(search_insert([1,3,5,6], 2))   # 1
print(search_insert([1,3,5,6], 7))   # 4
```

**Key change:** `hi = len(nums)` (not `len-1`) because the target could land after all elements.

---

## Q&A 3 — First Bad Version

**Q:** You have n versions; the first bad version causes all subsequent versions to be bad. Minimize API calls to `is_bad(version)`.

**A — Binary search on a boolean predicate:**
```python
def first_bad_version(n: int, is_bad) -> int:
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if is_bad(mid):
            hi = mid          # bad version could be mid itself
        else:
            lo = mid + 1      # mid is good, answer is strictly after
    return lo                 # lo == hi, converged to first bad
```

**Complexity:** O(log n) API calls

**Pattern recognition:** "Any time a problem has a monotone predicate (False...False...True...True) and you want the first True, use binary search on the predicate."

---

## Q&A 4 — Find Minimum in Rotated Sorted Array

**Q:** Given a rotated sorted array (e.g., `[4,5,6,7,0,1,2]`), find the minimum.

**A — Binary search on the rotation point:**
```python
def find_min(nums: list[int]) -> int:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            # minimum is in the right half
            lo = mid + 1
        else:
            # minimum is mid or in the left half
            hi = mid
    return nums[lo]

# Example
print(find_min([3,4,5,1,2]))   # 1
print(find_min([4,5,6,7,0,1,2]))  # 0
print(find_min([11,13,15,17]))    # 11
```

**Complexity:** Time O(log n), Space O(1)

**Explain the key comparison:** "I compare `nums[mid]` to `nums[hi]`. If mid is larger than hi, the minimum must be in the right portion. Otherwise it's mid or left."

---

## Q&A 5 — Kth Largest Element

**Q:** Find the kth largest element in an unsorted array.

**A — Multiple approaches:**

```python
import heapq, random

# Approach 1: Sort — O(n log n), simplest
def find_kth_largest_sort(nums, k):
    return sorted(nums, reverse=True)[k-1]

# Approach 2: Min-heap of size k — O(n log k)
def find_kth_largest_heap(nums, k):
    heap = []
    for n in nums:
        heapq.heappush(heap, n)
        if len(heap) > k:
            heapq.heappop(heap)     # pop the smallest; heap stays size k
    return heap[0]                  # smallest in heap = kth largest overall

# Approach 3: Quickselect — O(n) average, O(n²) worst
def find_kth_largest(nums, k):
    target = len(nums) - k         # kth largest = (n-k)th smallest (0-indexed)

    def quickselect(lo, hi):
        pivot = nums[hi]
        p = lo
        for i in range(lo, hi):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[hi] = nums[hi], nums[p]   # pivot in final position

        if p < target:   return quickselect(p+1, hi)
        elif p > target: return quickselect(lo, p-1)
        else:            return nums[p]

    return quickselect(0, len(nums)-1)

# Example
print(find_kth_largest([3,2,1,5,6,4], 2))   # 5
print(find_kth_largest([3,2,3,1,2,4,5,5,6], 4))  # 4
```

**When to use each:**
- "Can I sort?" → Sort, O(n log n), simplest
- "k is small" → Heap, O(n log k)
- "Optimal time" → Quickselect, O(n) average

---

## Q&A 6 — Interview Dialogue: Why Use Binary Search?

**Interviewer:** "When do you reach for binary search?"

**You:** "Binary search applies any time the search space is monotone — that is, you can eliminate half of it based on a condition. The classic case is a sorted array. But it also applies to:

1. **Sorted predicate problems** — `first_bad_version`, `find_first_true`
2. **Minimizing/maximizing a value** — 'what is the minimum capacity that works?' If I can answer 'does X work?' in O(n), I can binary search on X in O(n log(max_X))
3. **Rotated arrays** — modified binary search on the rotation point
4. **Answer-space binary search** — when the answer is a number in a range and you can verify a candidate answer efficiently"

---

## The "Binary Search on the Answer" Pattern

Many hard-looking problems reduce to: "Find the minimum X such that condition(X) is True."

```python
def solve(data, lo_bound, hi_bound):
    # Is 'capacity' sufficient? This is your custom check.
    def feasible(capacity):
        # Example: ship packages in <= D days with 'capacity' weight
        days, current = 1, 0
        for weight in data:
            if current + weight > capacity:
                days += 1
                current = 0
            current += weight
        return days <= D

    lo, hi = lo_bound, hi_bound
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid         # mid works, maybe smaller also works
        else:
            lo = mid + 1     # mid doesn't work, need bigger
    return lo
```

**Example problems that use this pattern:**
- Capacity To Ship Packages Within D Days
- Koko Eating Bananas
- Split Array Largest Sum

---

## Common Mistakes

1. **Infinite loop with `mid = (lo + hi) // 2` and `hi = mid - 1` forgotten** — always make sure lo/hi move
2. **Wrong boundary on hi** — `hi = n` vs `hi = n-1` depends on whether target can be at position n
3. **Using `while lo <= hi` vs `while lo < hi`** — learn both; which you pick must match your boundary updates
4. **Not sorting before binary search** — binary search requires sorted (or monotone) input
5. **Comparing wrong elements in rotated array** — compare to `nums[hi]` not `nums[lo]`
