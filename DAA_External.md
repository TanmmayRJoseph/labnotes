### Knapsack Problem

```py
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]],
                               dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7

print("Maximum value:", knapsack(weights, values, capacity))
```

---

### Dijkstra’s Algorithm

```py
import heapq

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist


graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}

result = dijkstra(graph, 'A')
for node, d in result.items():
    print(f"Shortest distance from A to {node}: {d}")
```

---

### Binary Search Recursive and Iterative

```py
def binary_search_iterative(arr, key):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def binary_search_recursive(arr, key, low, high):
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == key:
        return mid
    elif arr[mid] < key:
        return binary_search_recursive(arr, key, mid + 1, high)
    else:
        return binary_search_recursive(arr, key, low, mid - 1)


arr = [2, 5, 8, 12, 16, 23, 38, 56]
key = 23

print("Iterative: found at index", binary_search_iterative(arr, key))
print("Recursive: found at index", binary_search_recursive(arr, key, 0, len(arr) - 1))
```

---

### Merge Sort

```py
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result += left[i:]
    result += right[j:]
    return result


arr = [38, 27, 43, 3, 9, 82, 10]
print("Sorted array:", merge_sort(arr))
```

---

### Selection Sort

```py
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


arr = [64, 25, 12, 22, 11]
print("Sorted array:", selection_sort(arr))
```

---

### Minimum and Maximum in an Array

```py
def find_min_max(arr):
    minimum = maximum = arr[0]
    for num in arr:
        if num < minimum:
            minimum = num
        if num > maximum:
            maximum = num
    return minimum, maximum


arr = [34, 7, 23, 32, 5, 62]
minimum, maximum = find_min_max(arr)
print("Minimum:", minimum)
print("Maximum:", maximum)
```
