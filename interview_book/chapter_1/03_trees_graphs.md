# Chapter 1 — Section 3: Trees & Graphs

## Core Concepts

**Binary Tree node:**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

**Graph node (adjacency list — most common in interviews):**
```python
from collections import defaultdict
graph = defaultdict(list)  # node → [neighbors]
```

**The three tree traversals you must know cold:**
```
Inorder:   left → root → right   (gives sorted order in BST)
Preorder:  root → left → right   (good for copying/serializing)
Postorder: left → right → root   (good for deletion, height)
```

**BFS vs DFS decision rule:**
- BFS → shortest path, level-by-level
- DFS → path existence, exhaustive search, backtracking

---

## Q&A 1 — Maximum Depth of Binary Tree

**Q:** Find the maximum depth (number of nodes along the longest root-to-leaf path).

**A — Recursive DFS (most natural):**

```python
def max_depth(root: TreeNode | None) -> int:
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**A — Iterative BFS (level count):**
```python
from collections import deque

def max_depth_bfs(root: TreeNode | None) -> int:
    if not root:
        return 0
    depth = 0
    q = deque([root])
    while q:
        depth += 1
        for _ in range(len(q)):       # process one full level
            node = q.popleft()
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
    return depth
```

**Complexity:** Time O(n), Space O(h) recursive / O(w) iterative (w = max width).

**Say this:** "Recursion here reads like the definition of depth — the depth of a tree is one plus the max depth of its children. The base case is a null node which has depth zero."

---

## Q&A 2 — Invert Binary Tree

**Q:** Flip the tree — every left child becomes the right child and vice versa.

**A:**
```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

**Complexity:** Time O(n), Space O(h)

**Explain the swap:** "I swap the children first (via recursion returning the inverted subtrees), then return the current node. Python's tuple assignment means both sides evaluate before the swap."

---

## Q&A 3 — Symmetric Tree

**Q:** Check if a binary tree is a mirror of itself (symmetric around its center).

**A:**
```python
def is_symmetric(root: TreeNode | None) -> bool:
    def mirror(left, right) -> bool:
        if not left and not right: return True
        if not left or not right:  return False
        return (left.val == right.val and
                mirror(left.left, right.right) and
                mirror(left.right, right.left))

    return mirror(root, root)
```

**Key insight:** "Two trees are mirrors if their roots are equal, the left subtree of one is the mirror of the right subtree of the other, and vice versa."

---

## Q&A 4 — Path Sum

**Q:** Given a root and a `target_sum`, return `True` if there is a root-to-leaf path where the node values sum to `target_sum`.

**A:**
```python
def has_path_sum(root: TreeNode | None, target: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:     # leaf
        return root.val == target
    remaining = target - root.val
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)
```

**Complexity:** Time O(n), Space O(h)

**Pattern:** "Subtract-as-you-go. When you hit a leaf, check if the remaining sum is zero."

---

## Q&A 5 — Number of Islands (BFS/DFS on Grid)

**Q:** Given a 2D grid of '1' (land) and '0' (water), count the number of islands.

**A — DFS flood-fill:**
```python
def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'               # mark visited (mutate in-place)
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)              # sink the entire island

    return count

# Example
grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]
print(num_islands(grid))  # 3
```

**Complexity:** Time O(m·n), Space O(m·n) stack depth worst case

**Say this:** "Each cell is visited at most once. When I find a '1', I sink the entire connected island by marking cells '0' before recursing. If mutating input is not allowed, use a `visited` set."

---

## Q&A 6 — Level Order Traversal (BFS)

**Q:** Return the level-by-level values of a binary tree.

**A — The BFS level-snapshot pattern:**
```python
from collections import deque

def level_order(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []
    result = []
    q = deque([root])

    while q:
        level_size = len(q)          # snapshot: how many nodes at this level
        level = []
        for _ in range(level_size):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        result.append(level)

    return result
```

**Complexity:** Time O(n), Space O(w) where w is max width

**The `len(q)` snapshot trick** is the key to BFS level-by-level processing. Memorize this pattern.

---

## Q&A 7 — Interview Dialogue: When to Use BFS vs DFS?

**Interviewer:** "For Number of Islands, you used DFS. Why not BFS?"

**You:** "Either works for connectivity. I'd choose BFS when I need the shortest path — for example, if the question asked for the minimum steps to reach water from any land cell. DFS is often simpler to write recursively and has the same complexity here. BFS uses a queue and is naturally iterative, which avoids stack overflow on very deep graphs."

```python
# BFS version of num_islands for comparison
def num_islands_bfs(grid):
    rows, cols = len(grid), len(grid[0])
    count = 0

    def bfs(r, c):
        q = deque([(r, c)])
        grid[r][c] = '0'
        while q:
            row, col = q.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = row+dr, col+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    q.append((nr, nc))
                    grid[nr][nc] = '0'

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                bfs(r, c)
    return count
```

---

## Pattern Recognition Cheat Sheet

| Signal | Reach for |
|---|---|
| "shortest path" | BFS |
| "any path exists" | DFS |
| "process level by level" | BFS with level snapshot (`len(q)`) |
| "tree height / depth" | Recursive DFS |
| "BST — find value" | Binary search on BST property |
| "all paths / exhaustive" | DFS + backtrack |
| "connected components" | BFS/DFS flood fill, or Union-Find |

---

## Common Mistakes

1. **Forgetting the base case** — `if not root: return ...` before touching children
2. **Mutating grid without permission** — ask "can I modify the input?" If not, use `visited = set()`
3. **Using list as queue** — `list.pop(0)` is O(n); always use `collections.deque`
4. **Not handling disconnected graphs** — outer loop over all nodes, not just one start
