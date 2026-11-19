# AI & ML Lab Programs (Simple Python Code + Notes)

**Setup (run once):**
```bash
pip install numpy pandas scikit-learn
```

---

## 1. Breadth First Search (BFS) and Depth First Search (DFS)

**Important notes**
- **BFS** explores level by level. It uses a **queue** (FIFO). It finds the shortest path in an unweighted graph.
- **DFS** goes as deep as possible first. It uses a **stack** (or recursion). It uses less memory.
- Both keep a `visited` set so no node is visited twice.
- Time complexity: **O(V + E)** for both.

```python
from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

def bfs(start):
    visited = [start]
    queue = deque([start])
    while queue:
        node = queue.popleft()          # take from front
        print(node, end=" ")
        for nb in graph[node]:
            if nb not in visited:
                visited.append(nb)
                queue.append(nb)        # add to back

def dfs(node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node, end=" ")
    for nb in graph[node]:
        if nb not in visited:
            dfs(nb, visited)            # go deeper

print("BFS:", end=" "); bfs('A')
print("\nDFS:", end=" "); dfs('A')
```

**Output:**
```
BFS: A B C D E F
DFS: A B D E F C
```

---

## 2. Water Jug Problem

**Important notes**
- Two jugs with capacities (e.g., 4L and 3L). Goal: get exactly 2L in a jug.
- A **state** = `(water in jug1, water in jug2)`.
- Allowed moves: **fill** a jug, **empty** a jug, **pour** one jug into the other.
- We use **BFS** over states, so the answer is the shortest sequence of steps.
- A solution exists only if the target is a multiple of `gcd(cap1, cap2)` and is <= the larger jug.

```python
from collections import deque

def water_jug(cap1, cap2, target):
    start = (0, 0)
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (a, b), path = queue.popleft()
        if a == target or b == target:
            return path

        moves = [
            (cap1, b),                                  # fill jug1
            (a, cap2),                                  # fill jug2
            (0, b),                                     # empty jug1
            (a, 0),                                     # empty jug2
            (a - min(a, cap2 - b), b + min(a, cap2 - b)),   # pour 1 -> 2
            (a + min(b, cap1 - a), b - min(b, cap1 - a)),   # pour 2 -> 1
        ]
        for m in moves:
            if m not in visited:
                visited.add(m)
                queue.append((m, path + [m]))
    return None

solution = water_jug(4, 3, 2)
print("Steps (jug1, jug2):")
for step in solution:
    print(step)
```

---

## 3. Hill Climbing Algorithm

**Important notes**
- A **local search** algorithm: start somewhere, move to the best neighbour, stop when no neighbour is better.
- Keeps only the **current state** (no memory of the path), so it is very fast.
- **Problems:** can get stuck at a *local maximum*, a *plateau* (flat area), or a *ridge*.
- Fixes: random restart, simulated annealing.

```python
import random

def f(x):                       # function to maximize
    return -(x - 3) ** 2 + 9    # peak at x = 3, value 9

def hill_climbing(step=0.1, iterations=1000):
    current = random.uniform(-10, 10)       # random start
    for _ in range(iterations):
        left, right = current - step, current + step
        best = max([left, right, current], key=f)   # best neighbour
        if best == current:                 # no better neighbour -> stop
            break
        current = best
    return current

x = hill_climbing()
print("Best x =", round(x, 2))
print("Max value =", round(f(x), 2))
```

---

## 4. AO\* Algorithm

**Important notes**
- Used for **AND-OR graphs** (problems that split into sub-problems).
- **OR arc:** solve *any one* option. **AND arc:** solve *all* children together.
- Cost of a node = min over its options of `sum(child cost + edge cost)`.
- `H` holds the heuristic value of each leaf; non-leaf values get updated as we compute (this is the "revision" step).
- Below is a **simplified AO\*** (it evaluates the whole graph recursively and then prints the best solution graph).

```python
# Each node maps to a list of OPTIONS (OR).
# Each option is a list of children that must ALL be solved (AND).
graph = {
    'A': [['B'], ['C', 'D']],      # A = B   OR   (C AND D)
    'B': [['E'], ['F']],           # B = E   OR   F
    'C': [['G'], ['H', 'I']],      # C = G   OR   (H AND I)
    'D': [['J']],                  # D = J
}

# Heuristic cost of every node (leaves keep this value)
H = {'A': 0, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 4,
     'G': 5, 'H': 7, 'I': 7, 'J': 1}

EDGE_COST = 1
best_choice = {}

def ao_star(node):
    if node not in graph:               # leaf node
        return H[node]
    best_cost = float('inf')
    for option in graph[node]:
        cost = sum(ao_star(child) + EDGE_COST for child in option)
        if cost < best_cost:
            best_cost = cost
            best_choice[node] = option
    H[node] = best_cost                 # update (revise) node cost
    return best_cost

def print_solution(node):
    if node in best_choice:
        print(node, "->", " AND ".join(best_choice[node]))
        for child in best_choice[node]:
            print_solution(child)

print("Minimum cost of A =", ao_star('A'))
print("Best solution graph:")
print_solution('A')
```

---

## 5. Heart Disease Prediction using Logistic Regression

**Important notes**
- **Logistic Regression** is used for **binary classification** (disease = 1 / no disease = 0).
- It applies the **sigmoid** function to give a probability between 0 and 1. If probability >= 0.5, predict 1.
- Steps: load data -> split train/test -> scale features -> train -> predict -> evaluate.
- Dataset: the UCI / Kaggle **Heart Disease** CSV (`heart.csv`, target column `target`). Put it in the same folder.
- If `heart.csv` is not found, the code uses dummy data so it still runs.

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
if os.path.exists("heart.csv"):
    data = pd.read_csv("heart.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
else:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=13, random_state=1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict and evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

---

## 6. Diabetes Prediction using SVM and a Linear Classifier

**Important notes**
- **SVM** finds the line/hyperplane that separates the two classes with the **maximum margin**.
- **Kernel:** `linear` for straight boundaries, `rbf` for curved boundaries.
- **Linear classifier** here = `LinearSVC` (a fast linear SVM). It draws only a straight boundary.
- Feature **scaling is a must** for SVM.
- Dataset: **Pima Indians Diabetes** (`diabetes.csv`, target column `Outcome`). Dummy data is used if the file is missing.

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

# 1. Load data
if os.path.exists("diabetes.csv"):
    data = pd.read_csv("diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
else:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=8, random_state=1)

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. SVM (RBF kernel)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print("SVM (rbf) accuracy      :", accuracy_score(y_test, svm.predict(X_test)))

# 4. SVM with linear kernel
svm_lin = SVC(kernel='linear')
svm_lin.fit(X_train, y_train)
print("SVM (linear) accuracy   :", accuracy_score(y_test, svm_lin.predict(X_test)))

# 5. Linear classifier
lin = LinearSVC(max_iter=10000)
lin.fit(X_train, y_train)
print("Linear classifier acc.  :", accuracy_score(y_test, lin.predict(X_test)))
```

---

## 7. KNN and SVM for Classification

**Important notes**
- **KNN:** to classify a point, look at its **K nearest neighbours** and take a **majority vote**. No real training (a "lazy learner"). Needs feature scaling. K is usually odd (3, 5, 7).
- **SVM:** finds the best separating boundary (max margin).
- We use the **Iris dataset** (built into sklearn, so no download needed): 3 flower classes, 4 features.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN accuracy:", accuracy_score(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))

# 4. SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM accuracy:", accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
```

---

## 8. Artificial Neural Network with Backpropagation

**Important notes**
- Network layers: **Input -> Hidden -> Output**. Here: 4 inputs -> 8 hidden -> 3 outputs (Iris).
- **Forward pass:** `output = sigmoid(input x weights + bias)`, layer by layer.
- **Backpropagation:** 
  1. Find the error at the output: `target - output`.
  2. Multiply by the sigmoid derivative `out * (1 - out)` to get the delta.
  3. Send delta backward to the hidden layer using the weights.
  4. Update weights: `W = W + learning_rate * input.T x delta`.
- Repeat for many **epochs** until the error is small.
- Test on data the network has not seen (test set).
- Code uses only **NumPy** (no deep learning library), so every step is visible.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

# 1. Data
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = np.eye(3)[iris.target]                  # one-hot encode labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(out):                # 'out' is already sigmoid(x)
    return out * (1 - out)

# 3. Initialize weights (4 -> 8 -> 3)
W1 = np.random.randn(4, 8) * 0.5
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 3) * 0.5
b2 = np.zeros((1, 3))
lr = 0.5

# 4. Train
for epoch in range(3000):
    # Forward pass
    hidden = sigmoid(X_train @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)

    # Backward pass (backpropagation)
    error = y_train - output
    d_output = error * sigmoid_derivative(output)
    d_hidden = (d_output @ W2.T) * sigmoid_derivative(hidden)

    # Update weights
    W2 += lr * hidden.T @ d_output / len(X_train)
    b2 += lr * d_output.mean(axis=0, keepdims=True)
    W1 += lr * X_train.T @ d_hidden / len(X_train)
    b1 += lr * d_hidden.mean(axis=0, keepdims=True)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(error ** 2):.4f}")

# 5. Test
hidden = sigmoid(X_test @ W1 + b1)
output = sigmoid(hidden @ W2 + b2)
pred = np.argmax(output, axis=1)
actual = np.argmax(y_test, axis=1)
print("Test Accuracy:", np.mean(pred == actual))
```

---

## Quick Summary Table

| # | Topic | Type | Key Idea |
|---|-------|------|----------|
| 1 | BFS / DFS | Uninformed search | Queue (BFS) vs Stack (DFS) |
| 2 | Water Jug | State-space search | BFS over (jug1, jug2) states |
| 3 | Hill Climbing | Local search | Always move to the better neighbour |
| 4 | AO\* | Informed search (AND-OR) | Min cost with AND/OR options |
| 5 | Heart Disease | Logistic Regression | Sigmoid -> probability -> class |
| 6 | Diabetes | SVM + Linear classifier | Max-margin boundary |
| 7 | KNN & SVM | Classification | Majority vote vs max margin |
| 8 | ANN | Neural network | Forward pass + backpropagation |
