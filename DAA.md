# Design and Analysis of Algorithms

Algorithms covered: Bucket Sort, Merge Sort, Binary Search, Counting Sort, Radix Sort, Minimum and Maximum, Quick Sort.

---

## Complexity Summary

| Algorithm | Best | Average | Worst | Space | Stable | Approach |
|-----------|------|---------|-------|-------|--------|----------|
| Bucket Sort | O(n + k) | O(n + k) | O(n²) | O(n + k) | Yes | Distribution |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | Divide and Conquer |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) iterative | – | Divide and Conquer |
| Counting Sort | O(n + k) | O(n + k) | O(n + k) | O(n + k) | Yes | Non-comparison |
| Radix Sort | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) | Yes | Non-comparison |
| Min and Max (D&C) | O(n) | O(n) | O(n) | O(log n) recursion | – | Divide and Conquer |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Divide and Conquer |

(k = range of values or number of buckets, d = number of digits)

---

## 1. Bucket Sort

### Idea
Distribute the elements into several buckets, sort each bucket (usually with insertion sort), then concatenate the buckets. Works best when input is uniformly distributed over a range, e.g. floats in [0, 1).

### Algorithm
1. Create n empty buckets.
2. Put each element `x` into bucket `int(n * x)`.
3. Sort every bucket individually.
4. Concatenate all buckets in order.

### Code
```python
def insertion_sort(bucket):
    for i in range(1, len(bucket)):
        key = bucket[i]
        j = i - 1
        while j >= 0 and bucket[j] > key:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = key
    return bucket


def bucket_sort(arr):
    n = len(arr)
    buckets = [[] for _ in range(n)]

    # Distribute elements into buckets (values in [0, 1))
    for x in arr:
        index = int(n * x)
        buckets[index].append(x)

    # Sort each bucket
    for b in buckets:
        insertion_sort(b)

    # Concatenate
    result = []
    for b in buckets:
        result.extend(b)
    return result


arr = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]
print("Sorted:", bucket_sort(arr))
```

### Output
```
Sorted: [0.12, 0.17, 0.21, 0.23, 0.26, 0.39, 0.68, 0.72, 0.78, 0.94]
```

### Complexity
- Best/Average: O(n + k) when elements are evenly spread.
- Worst: O(n²) when all elements land in one bucket.

---

## 2. Merge Sort

### Idea
Divide the array into two halves, recursively sort each half, then merge the two sorted halves.

### Recurrence
`T(n) = 2T(n/2) + O(n)` which gives `T(n) = O(n log n)` (Master Theorem, case 2).

### Code
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:      # <= keeps the sort stable
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


arr = [38, 27, 43, 3, 9, 82, 10]
print("Sorted:", merge_sort(arr))
```

### Output
```
Sorted: [3, 9, 10, 27, 38, 43, 82]
```

### Complexity
- Time: O(n log n) in all cases.
- Space: O(n) for the temporary arrays.

---

## 3. Binary Search

### Idea
Works only on a **sorted** array. Compare the target with the middle element and discard half of the array each step.

### Recurrence
`T(n) = T(n/2) + O(1)` which gives `T(n) = O(log n)`.

### Code (Iterative)
```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print("Index of 23:", binary_search(arr, 23))
print("Index of 40:", binary_search(arr, 40))
```

### Code (Recursive)
```python
def binary_search_rec(arr, target, low, high):
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_rec(arr, target, mid + 1, high)
    else:
        return binary_search_rec(arr, target, low, mid - 1)


arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search_rec(arr, 23, 0, len(arr) - 1))
```

### Output
```
Index of 23: 5
Index of 40: -1
```

### Complexity
- Best: O(1), Average/Worst: O(log n).
- Space: O(1) iterative, O(log n) recursive.

---

## 4. Counting Sort

### Idea
Non-comparison sort for integers in a known range `0..k`. Count how many times each value occurs, compute cumulative counts to find final positions, then place elements (traversing from the end to keep it stable).

### Code
```python
def counting_sort(arr):
    if not arr:
        return arr

    k = max(arr)
    count = [0] * (k + 1)

    # Step 1: count occurrences
    for x in arr:
        count[x] += 1

    # Step 2: cumulative count (gives final positions)
    for i in range(1, k + 1):
        count[i] += count[i - 1]

    # Step 3: build output array from the end (stable)
    output = [0] * len(arr)
    for x in reversed(arr):
        count[x] -= 1
        output[count[x]] = x

    return output


arr = [4, 2, 2, 8, 3, 3, 1]
print("Sorted:", counting_sort(arr))
```

### Output
```
Sorted: [1, 2, 2, 3, 3, 4, 8]
```

### Complexity
- Time: O(n + k), Space: O(n + k).
- Not suitable when `k` is much larger than `n`.
- Works only on non-negative integers (negatives need an offset).

---

## 5. Radix Sort

### Idea
Sort numbers digit by digit, starting from the least significant digit (LSD), using a **stable** sort (counting sort) for each digit.

### Code
```python
def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for x in arr:
        digit = (x // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for x in reversed(arr):
        digit = (x // exp) % 10
        count[digit] -= 1
        output[count[digit]] = x

    return output


def radix_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr


arr = [170, 45, 75, 90, 802, 24, 2, 66]
print("Sorted:", radix_sort(arr))
```

### Output
```
Sorted: [2, 24, 45, 66, 75, 90, 170, 802]
```

### Complexity
- Time: O(d × (n + k)) with k = 10 for decimal digits.
- Space: O(n + k).

---

## 6. Minimum and Maximum (Divide and Conquer)

### Idea
Split the array into two halves, find min and max of each half recursively, then compare the results. This uses about **3n/2 − 2** comparisons, fewer than the 2(n − 1) of the naive approach.

### Recurrence
`T(n) = 2T(n/2) + 2`, with `T(1) = 0` and `T(2) = 1`, which gives `T(n) = 3n/2 − 2` when n is a power of 2.

### Code
```python
def min_max(arr, low, high):
    # Only one element
    if low == high:
        return arr[low], arr[low]

    # Two elements
    if high == low + 1:
        if arr[low] < arr[high]:
            return arr[low], arr[high]
        else:
            return arr[high], arr[low]

    # More than two elements: divide
    mid = (low + high) // 2
    min1, max1 = min_max(arr, low, mid)
    min2, max2 = min_max(arr, mid + 1, high)

    return min(min1, min2), max(max1, max2)


arr = [22, 13, -5, -8, 15, 60, 17, 31, 47]
minimum, maximum = min_max(arr, 0, len(arr) - 1)
print("Minimum:", minimum)
print("Maximum:", maximum)
```

### Output
```
Minimum: -8
Maximum: 60
```

### Complexity
- Time: O(n), Space: O(log n) for the recursion stack.

---

## 7. Quick Sort

### Idea
Pick a **pivot**, partition the array so smaller elements go left of the pivot and larger go right, then recursively sort both parts.

### Recurrence
- Best/Average: `T(n) = 2T(n/2) + O(n)` which gives O(n log n).
- Worst (already sorted input with last element as pivot): `T(n) = T(n − 1) + O(n)` which gives O(n²).

### Code (Lomuto Partition)
```python
def partition(arr, low, high):
    pivot = arr[high]          # last element as pivot
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quick_sort(arr, low, high):
    if low < high:
        p = partition(arr, low, high)
        quick_sort(arr, low, p - 1)
        quick_sort(arr, p + 1, high)


arr = [10, 7, 8, 9, 1, 5]
quick_sort(arr, 0, len(arr) - 1)
print("Sorted:", arr)
```

### Output
```
Sorted: [1, 5, 7, 8, 9, 10]
```

### Randomized Quick Sort (avoids the worst case)
```python
import random

def randomized_partition(arr, low, high):
    r = random.randint(low, high)
    arr[r], arr[high] = arr[high], arr[r]
    return partition(arr, low, high)


def randomized_quick_sort(arr, low, high):
    if low < high:
        p = randomized_partition(arr, low, high)
        randomized_quick_sort(arr, low, p - 1)
        randomized_quick_sort(arr, p + 1, high)


arr = [10, 7, 8, 9, 1, 5]
randomized_quick_sort(arr, 0, len(arr) - 1)
print("Sorted:", arr)
```

### Complexity
- Best/Average: O(n log n), Worst: O(n²).
- Space: O(log n) average for the recursion stack.
- In-place but **not stable**.

---

## Quick Revision Points

- **Bucket**: distribute, sort each bucket, concatenate. Needs uniform data.
- **Merge**: divide, sort halves, merge. Always O(n log n), stable, needs O(n) extra space.
- **Binary Search**: sorted array only, halve the search space, O(log n).
- **Counting**: count occurrences, cumulative sum, place from the end. O(n + k).
- **Radix**: counting sort applied digit by digit from the least significant digit. O(d(n + k)).
- **Min and Max**: divide and conquer uses 3n/2 − 2 comparisons.
- **Quick**: pivot partition, O(n log n) average, O(n²) worst, in-place and unstable.
- Comparison-based sorting has a lower bound of Ω(n log n); counting, radix and bucket sort beat it by not comparing elements.