# Chapter 2 — Pattern 2: Two Pointers

## The Pattern

Two pointers work on a sorted (or sortable) structure where you can eliminate possibilities by moving pointers toward each other based on a comparison. Converts O(n²) brute force into O(n).

**Two variants:**
- **Converging (opposite ends):** `left = 0, right = n-1` — move inward. Used for pairs/triplets summing to target.
- **Same direction (fast/slow):** both start at left and move right at different speeds. Used for cycle detection, removing duplicates, finding middle.

---

## Q&A 1 — Two Sum II (Sorted Array)

**Q:** Given a sorted array, find two numbers that add up to target. Return 1-indexed positions.

**A:**
```python
def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    left, right = 0, len(numbers) - 1
    while left < right:
        total = numbers[left] + numbers[right]
        if total == target:
            return [left + 1, right + 1]    # 1-indexed
        elif total < target:
            left += 1                        # need bigger sum
        else:
            right -= 1                       # need smaller sum
    return []

# Example
print(two_sum_sorted([2, 7, 11, 15], 9))   # [1, 2]
print(two_sum_sorted([2, 3, 4], 6))        # [1, 3]
```

**Complexity:** Time O(n), Space O(1)

**Explain why it works:** "If `sum < target`, the left element is too small — moving left right can only increase the sum. If `sum > target`, the right element is too large — moving right left can only decrease it. We never miss the solution."

---

## Q&A 2 — 3Sum

**Q:** Find all unique triplets in an unsorted array that sum to zero.

**A — Sort first, then fix one element and use two pointers:**

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for the fixed element
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1
        target = -nums[i]

        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates for the pair
                while left < right and nums[left] == nums[left+1]:  left  += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left  += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1

    return result

# Example
print(three_sum([-1,0,1,2,-1,-4]))   # [[-1,-1,2],[-1,0,1]]
print(three_sum([0,1,1]))            # []
print(three_sum([0,0,0]))            # [[0,0,0]]
```

**Complexity:** Time O(n²), Space O(1) (excluding output)

**Duplicate skipping logic:**
> "After sort, duplicates are adjacent. I skip duplicate `i` values at the outer loop start. After finding a valid triplet, I skip duplicate `left` and `right` values before moving the pointers inward."

---

## Q&A 3 — Container With Most Water

**Q:** Given an array where `height[i]` is the height of a vertical line, find two lines that form a container holding the most water.

**A:**
```python
def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    best = 0

    while left < right:
        h = min(height[left], height[right])
        w = right - left
        best = max(best, h * w)

        # Move the shorter side — moving the taller side can't help
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return best

# Example
print(max_area([1,8,6,2,5,4,8,3,7]))  # 49
print(max_area([1,1]))                  # 1
```

**Complexity:** Time O(n), Space O(1)

**Explain the greedy choice:** "The area is limited by the shorter line. If I move the taller line inward, the width decreases and the height can only stay the same or get shorter — area can't improve. So I always move the shorter line."

---

## Q&A 4 — Remove Duplicates from Sorted Array

**Q:** Remove duplicates in-place from a sorted array and return the count of unique elements.

**A — Fast/slow pointers:**
```python
def remove_duplicates(nums: list[int]) -> int:
    if not nums:
        return 0
    slow = 0    # points to last unique element written
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

# Example
nums = [1,1,2]
k = remove_duplicates(nums)
print(k, nums[:k])     # 2  [1, 2]

nums = [0,0,1,1,1,2,2,3,3,4]
k = remove_duplicates(nums)
print(k, nums[:k])     # 5  [0, 1, 2, 3, 4]
```

**Pattern:** "Slow pointer marks where to write next unique element. Fast pointer scans ahead. When fast finds something different from slow, write it at slow+1."

---

## Q&A 5 — Trapping Rain Water

**Q:** Given an elevation map, compute how much water it can trap.

**A — Two pointer, track max from each side:**
```python
def trap(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]  # trapped by left_max
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]  # trapped by right_max
            right -= 1

    return water

# Example
print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
print(trap([4,2,0,3,2,5]))              # 9
```

**Complexity:** Time O(n), Space O(1)

**Why it works:** "Water trapped at any position equals `min(max_left, max_right) - height[i]`. The two-pointer approach ensures we always know which side's max is the binding constraint — we process the side with the smaller current max."

---

## Q&A 6 — Valid Palindrome

**Q:** A string is a palindrome (ignoring non-alphanumeric characters, case-insensitive). Check if it is.

**A:**
```python
def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

# Example
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))                      # False
```

**Complexity:** Time O(n), Space O(1)

---

## Q&A 7 — Linked List Cycle (Fast/Slow Pointers)

**Q:** Detect if a linked list has a cycle.

**A — Floyd's Tortoise and Hare:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode | None) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:         # same object in memory
            return True
    return False
```

**Extension — Find cycle start:**
```python
def detect_cycle_start(head: ListNode | None) -> ListNode | None:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # Move one pointer to head; advance both at speed 1
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow    # cycle start
    return None
```

**Complexity:** Time O(n), Space O(1)

---

## Pattern Recognition Cheat Sheet

| Signal | Two-pointer type |
|---|---|
| Sorted array, find pair/triplet summing to X | Converging |
| Palindrome check | Converging |
| Maximize/minimize area or water | Converging, greedy |
| Remove duplicates in-place | Fast/slow, same direction |
| Cycle detection in linked list | Fast/slow (Floyd's) |
| Middle of linked list | Fast/slow (stop when fast reaches end) |
| Partition array (quicksort) | Converging or Lomuto |

---

## Common Mistakes

1. **Forgetting to skip duplicates in 3Sum** — leads to duplicate triplets in output
2. **Using `==` instead of `is` for node comparison** — two different nodes can have the same value; use `is` for identity
3. **Off-by-one in `while left < right`** — use `<` not `<=` to avoid checking same element against itself
4. **Not sorting first for 3Sum/similar** — two pointers only work on sorted data
