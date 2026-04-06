# Chapter 2 — Pattern 3: BFS & DFS (Medium Problems)

## Core Templates (Memorize These)

**BFS — Queue-based, level-by-level:**
```python
from collections import deque

def bfs(start, goal, graph):
    visited = {start}
    queue = deque([start])          # (node, distance) if tracking distance

    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
```

**DFS — Recursive or stack-based:**
```python
def dfs(node, visited, graph):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited, graph)
```

---

## Q&A 1 — Clone Graph

**Q:** Given a reference to a node in a connected undirected graph, return a deep copy.

**A — BFS with a mapping dict:**
```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors or []

def clone_graph(node: Node | None) -> Node | None:
    if not node:
        return None

    clones = {}             # original node → its clone
    queue = deque([node])
    clones[node] = Node(node.val)

    while queue:
        curr = queue.popleft()
        for neighbor in curr.neighbors:
            if neighbor not in clones:
                clones[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            clones[curr].neighbors.append(clones[neighbor])

    return clones[node]
```

**Complexity:** Time O(V+E), Space O(V)

**Key insight:** "I use a dict to map each original node to its clone. This also serves as the visited set — if the original is already in the dict, I've already created its clone."

---

## Q&A 2 — Course Schedule (Cycle Detection)

**Q:** There are n courses. Given prerequisites `[a, b]` meaning 'take b before a', can you finish all courses?

**A — This is cycle detection in a directed graph (topological sort):**

```python
from collections import deque

def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    # Build adjacency list and in-degree count
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Kahn's algorithm: start with courses that have no prerequisites
    queue = deque([c for c in range(num_courses) if in_degree[c] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses    # if < num_courses, there's a cycle

# Example
print(can_finish(2, [[1,0]]))          # True  (0 → 1)
print(can_finish(2, [[1,0],[0,1]]))    # False (cycle)
```

**Complexity:** Time O(V+E), Space O(V+E)

**Alternative DFS approach with coloring:**
```python
def can_finish_dfs(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 0=unvisited, 1=in progress (on stack), 2=done
    color = [0] * num_courses

    def has_cycle(node):
        if color[node] == 1: return True    # back edge = cycle
        if color[node] == 2: return False   # already fully processed
        color[node] = 1
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        color[node] = 2
        return False

    return not any(has_cycle(c) for c in range(num_courses) if color[c] == 0)
```

---

## Q&A 3 — Word Ladder

**Q:** Given `beginWord`, `endWord`, and a word list, find the shortest transformation sequence length where each step changes exactly one letter and each intermediate word must be in the word list.

**A — BFS on word graph (shortest path):**

```python
from collections import deque

def ladder_length(begin: str, end: str, word_list: list[str]) -> int:
    word_set = set(word_list)
    if end not in word_set:
        return 0

    queue = deque([(begin, 1)])   # (word, steps)
    visited = {begin}

    while queue:
        word, steps = queue.popleft()

        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]

                if next_word == end:
                    return steps + 1

                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))

    return 0

# Example
print(ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"]))  # 5
# hit → hot → dot → dog → cog
```

**Complexity:** Time O(n · L · 26) where L is word length, Space O(n · L)

**Why BFS and not DFS:** "BFS finds the shortest path by definition — it explores all paths of length 1, then all of length 2, etc. The first time we reach `endWord` is the shortest path."

---

## Q&A 4 — Pacific Atlantic Water Flow

**Q:** A grid represents land elevation. Water flows to adjacent cells with equal or lower height. Find all cells from which water can flow to both the Pacific Ocean (top-left border) and Atlantic Ocean (bottom-right border).

**A — Reverse BFS from each ocean's border:**

```python
from collections import deque

def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    rows, cols = len(heights), len(heights[0])
    DIRS = [(0,1),(0,-1),(1,0),(-1,0)]

    def bfs(starts):
        visited = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in visited
                        and heights[nr][nc] >= heights[r][c]):  # water flows UP in reverse
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    pacific_starts = [(0, c) for c in range(cols)] + [(r, 0) for r in range(rows)]
    atlantic_starts = [(rows-1, c) for c in range(cols)] + [(r, cols-1) for r in range(rows)]

    pacific = bfs(pacific_starts)
    atlantic = bfs(atlantic_starts)

    return [[r, c] for r, c in pacific & atlantic]
```

**Complexity:** Time O(m·n), Space O(m·n)

**Explain the reversal trick:** "Instead of simulating water flowing down from every cell (which would be O(n²) naive), I reverse the problem. I do BFS from each ocean's border and find which cells water can reach going uphill. The intersection gives cells that can drain to both."

---

## Q&A 5 — Rotting Oranges (Multi-Source BFS)

**Q:** Grid has 0 (empty), 1 (fresh), 2 (rotten). Each minute, rotten oranges rot adjacent fresh ones. Return minimum minutes until no fresh oranges remain (-1 if impossible).

**A — Multi-source BFS starting from all rotten oranges simultaneously:**

```python
from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))    # (row, col, time)
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    time = 0
    DIRS = [(0,1),(0,-1),(1,0),(-1,0)]

    while queue:
        r, c, t = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                time = t + 1
                queue.append((nr, nc, t + 1))

    return time if fresh == 0 else -1

# Example
print(oranges_rotting([[2,1,1],[1,1,0],[0,1,1]]))  # 4
print(oranges_rotting([[2,1,1],[0,1,1],[1,0,1]]))  # -1
```

**Multi-source BFS key idea:** "I seed the queue with ALL rotten oranges at time 0. BFS then naturally spreads the rot in waves — all oranges at distance 1 rot at time 1, distance 2 at time 2, etc."

---

## Q&A 6 — Interview Dialogue: BFS/DFS Trade-offs for Graph Problems

**Interviewer:** "You keep using BFS for 'shortest path' problems. Can you always use DFS instead?"

**You:** "DFS can tell you whether a path exists, but it may find a longer path first. For unweighted graphs, BFS guarantees the shortest path because it explores by distance layers. DFS explores depth-first and might find a path of length 100 before finding the length-3 path.

For weighted graphs, you'd use Dijkstra's (not tested at screening). For unweighted, always reach for BFS when the problem asks for 'minimum steps', 'shortest', or 'fewest operations'.

DFS is better for: cycle detection (using a visited coloring scheme), topological sort, connected components, and exhaustive search with backtracking."

---

## Pattern Recognition Cheat Sheet

| Problem | Algorithm | Why |
|---|---|---|
| Shortest path, unweighted | BFS | Explores by distance layers |
| Connected components | Either | DFS simpler to write recursively |
| Cycle detection, undirected | BFS or DFS with parent tracking | Check if neighbor already visited (not parent) |
| Cycle detection, directed | DFS with color (0/1/2) | Detect back edges |
| Topological sort | BFS Kahn's or DFS postorder | Process nodes with no in-degree first |
| Multi-source shortest path | Multi-source BFS | Seed queue with all sources |
| All paths | DFS + backtracking | Need exhaustive enumeration |

---

## Common Mistakes

1. **Marking visited too late in BFS** — mark as visited when you enqueue (not when you dequeue) to avoid adding the same node multiple times
2. **Using a list instead of deque** — `list.pop(0)` is O(n); `deque.popleft()` is O(1)
3. **Not handling disconnected graphs** — outer loop over all nodes when graph may not be connected
4. **Confusing directed vs undirected cycle detection** — for directed graphs, use 3-color DFS; for undirected, just track parent
