# Statistics Practical Programs

### Installation
```bash
pip install numpy scipy matplotlib
```

---

## 1. Range, Variance and Standard Deviation

**Data:** 72, 88, 80, 96, 85

### Formulas
- Range = Maximum − Minimum
- Mean (x̄) = Σx / n
- Population Variance (σ²) = Σ(x − x̄)² / n
- Sample Variance (s²) = Σ(x − x̄)² / (n − 1)
- Standard Deviation = √Variance

### Manual Calculation
- Mean = (72 + 88 + 80 + 96 + 85) / 5 = 421 / 5 = **84.2**
- Range = 96 − 72 = **24**
- Σ(x − x̄)² = 148.84 + 14.44 + 17.64 + 139.24 + 0.64 = **320.8**
- Population Variance = 320.8 / 5 = **64.16**, SD = **8.01**
- Sample Variance = 320.8 / 4 = **80.2**, SD = **8.9554**

### Code
```python
import math
import statistics

data = [72, 88, 80, 96, 85]
n = len(data)

# Range
data_range = max(data) - min(data)

# Mean
mean = sum(data) / n

# Sum of squared deviations
ss = sum((x - mean) ** 2 for x in data)

# Population variance and standard deviation
pop_var = ss / n
pop_sd = math.sqrt(pop_var)

# Sample variance and standard deviation
samp_var = ss / (n - 1)
samp_sd = math.sqrt(samp_var)

print("Data               :", data)
print("Mean               :", mean)
print("Range              :", data_range)
print("Population Variance:", round(pop_var, 4))
print("Population Std Dev :", round(pop_sd, 4))
print("Sample Variance    :", round(samp_var, 4))
print("Sample Std Dev     :", round(samp_sd, 4))

# Verification using the statistics module
print("\nUsing statistics module:")
print("pvariance:", statistics.pvariance(data))
print("pstdev   :", round(statistics.pstdev(data), 4))
print("variance :", statistics.variance(data))
print("stdev    :", round(statistics.stdev(data), 4))
```

### Output
```
Data               : [72, 88, 80, 96, 85]
Mean               : 84.2
Range              : 24
Population Variance: 64.16
Population Std Dev : 8.01
Sample Variance    : 80.2
Sample Std Dev     : 8.9554

Using statistics module:
pvariance: 64.16
pstdev   : 8.01
variance : 80.2
stdev    : 8.9554
```

---

## 2. Outlier Detection using Box Plot and Whiskers

### Theory
A box plot shows the five-number summary: minimum, Q1, median, Q3 and maximum.

- IQR = Q3 − Q1
- Lower fence = Q1 − 1.5 × IQR
- Upper fence = Q3 + 1.5 × IQR
- **Whiskers** extend to the smallest and largest values that lie inside the fences.
- Any value outside the fences is an **outlier** (drawn as a separate point).

### Code
```python
import numpy as np
import matplotlib.pyplot as plt

data = [10, 12, 14, 15, 18, 21, 22, 24, 26, 28, 95]

# Quartiles and IQR
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
median = np.median(data)
iqr = q3 - q1

# Fences
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

# Outliers
outliers = [x for x in data if x < lower_fence or x > upper_fence]

# Whisker ends (extreme values inside the fences)
inside = [x for x in data if lower_fence <= x <= upper_fence]
lower_whisker = min(inside)
upper_whisker = max(inside)

print("Q1             :", q1)
print("Median         :", median)
print("Q3             :", q3)
print("IQR            :", iqr)
print("Lower Fence    :", lower_fence)
print("Upper Fence    :", upper_fence)
print("Lower Whisker  :", lower_whisker)
print("Upper Whisker  :", upper_whisker)
print("Outliers       :", outliers)

# Box plot
plt.figure(figsize=(7, 4))
plt.boxplot(data, vert=False, whis=1.5)
plt.title("Box Plot with Whiskers and Outliers")
plt.xlabel("Values")
plt.show()
```

### Output
```
Q1             : 14.5
Median         : 21.0
Q3             : 25.0
IQR            : 10.5
Lower Fence    : -1.25
Upper Fence    : 40.75
Lower Whisker  : 10
Upper Whisker  : 28
Outliers       : [95]
```

95 lies above the upper fence (40.75), so it is the outlier.

---

## 3. Normal Probability Distribution

### Theory
The normal distribution is a symmetric, bell-shaped distribution defined by mean (μ) and standard deviation (σ).

- PDF: f(x) = (1 / (σ√(2π))) × e^(−(x − μ)² / (2σ²))
- Z-score: z = (x − μ) / σ
- Empirical rule: about 68% of values lie within 1σ, 95% within 2σ and 99.7% within 3σ.

### Code
```python
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 50        # mean
sigma = 10     # standard deviation

# PDF from the formula
def normal_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

print("PDF at x = 50 (formula):", round(normal_pdf(50, mu, sigma), 4))
print("PDF at x = 50 (scipy)  :", round(norm.pdf(50, mu, sigma), 4))

# Probabilities
print("P(X < 60)       :", round(norm.cdf(60, mu, sigma), 4))
print("P(40 < X < 60)  :", round(norm.cdf(60, mu, sigma) - norm.cdf(40, mu, sigma), 4))
print("P(X > 65)       :", round(1 - norm.cdf(65, mu, sigma), 4))

# Z-score
x = 65
z = (x - mu) / sigma
print("Z-score of 65   :", z)

# Value for a given probability (inverse CDF)
print("95th percentile :", round(norm.ppf(0.95, mu, sigma), 4))

# Plot
x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
y_vals = norm.pdf(x_vals, mu, sigma)

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, label=f"N({mu}, {sigma}²)")
plt.fill_between(x_vals, y_vals, where=(x_vals >= 40) & (x_vals <= 60),
                 alpha=0.3, label="P(40 < X < 60)")
plt.axvline(mu, linestyle="--", color="red", label="Mean")
plt.title("Normal Probability Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
```

### Output
```
PDF at x = 50 (formula): 0.0399
PDF at x = 50 (scipy)  : 0.0399
P(X < 60)       : 0.8413
P(40 < X < 60)  : 0.6827
P(X > 65)       : 0.0668
Z-score of 65   : 1.5
95th percentile : 66.4485
```

---

## 4. Discrete and Continuous Random Variables

### Theory
| | Discrete Random Variable | Continuous Random Variable |
|---|---|---|
| Values | Countable (0, 1, 2, ...) | Any value in an interval |
| Described by | PMF: P(X = x) | PDF: f(x) |
| Probability | P(X = x) is positive | P(X = x) = 0; use area under the curve |
| Total | Σ P(x) = 1 | ∫ f(x) dx = 1 |
| Mean | E[X] = Σ x·P(x) | E[X] = ∫ x·f(x) dx |
| Example | Roll of a die, Binomial | Height, Uniform, Normal |

### Part A: Discrete Random Variable (Fair Die and Binomial)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# ---- Fair die: X = outcome of one roll ----
values = [1, 2, 3, 4, 5, 6]
probs = [1 / 6] * 6                      # PMF

print("Value  P(X = x)")
for v, p in zip(values, probs):
    print(f"{v:<6} {p:.4f}")

mean = sum(v * p for v, p in zip(values, probs))
variance = sum((v - mean) ** 2 * p for v, p in zip(values, probs))

print("\nSum of probabilities:", round(sum(probs), 4))
print("E[X]  :", round(mean, 4))
print("Var(X):", round(variance, 4))
print("SD    :", round(variance ** 0.5, 4))

# Simulation of 1000 rolls
np.random.seed(42)
rolls = np.random.choice(values, size=1000, p=probs)
print("Simulated mean of 1000 rolls:", rolls.mean())

# ---- Binomial: n = 10 trials, p = 0.5 ----
n, p = 10, 0.5
print("\nBinomial(n=10, p=0.5)")
print("P(X = 5)  :", round(binom.pmf(5, n, p), 4))
print("P(X <= 3) :", round(binom.cdf(3, n, p), 4))
print("Mean      :", binom.mean(n, p))
print("Variance  :", binom.var(n, p))

# PMF bar plot
k = np.arange(0, n + 1)
plt.bar(k, binom.pmf(k, n, p))
plt.title("Discrete Random Variable: Binomial(10, 0.5) PMF")
plt.xlabel("Number of successes")
plt.ylabel("P(X = k)")
plt.show()
```

### Output (Part A)
```
Value  P(X = x)
1      0.1667
2      0.1667
3      0.1667
4      0.1667
5      0.1667
6      0.1667

Sum of probabilities: 1.0
E[X]  : 3.5
Var(X): 2.9167
SD    : 1.7078
Simulated mean of 1000 rolls: (close to 3.5)

Binomial(n=10, p=0.5)
P(X = 5)  : 0.2461
P(X <= 3) : 0.1719
Mean      : 5.0
Variance  : 2.5
```

### Part B: Continuous Random Variable (Uniform and Normal)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm

# ---- Uniform distribution on [0, 10] ----
a, b = 0, 10
dist = uniform(loc=a, scale=b - a)

print("Uniform(0, 10)")
print("PDF f(x) = 1/(b-a) :", dist.pdf(5))
print("P(X = 5) (a single point) : 0")
print("P(2 < X < 5)       :", round(dist.cdf(5) - dist.cdf(2), 4))
print("Mean               :", dist.mean())
print("Variance           :", round(dist.var(), 4))

# Simulation
np.random.seed(42)
samples = np.random.uniform(a, b, 1000)
print("Simulated mean     :", round(samples.mean(), 2), "(close to 5)")

# ---- Normal distribution as another continuous variable ----
print("\nNormal(0, 1)")
print("P(-1 < Z < 1):", round(norm.cdf(1) - norm.cdf(-1), 4))

# Plots
x = np.linspace(-2, 12, 500)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(x, dist.pdf(x))
ax[0].fill_between(x, dist.pdf(x), where=(x >= 2) & (x <= 5), alpha=0.3)
ax[0].set_title("Uniform(0, 10) PDF: P(2 < X < 5)")

ax[1].hist(samples, bins=20, density=True, edgecolor="black")
ax[1].set_title("Histogram of 1000 simulated values")

plt.tight_layout()
plt.show()
```

### Output (Part B)
```
Uniform(0, 10)
PDF f(x) = 1/(b-a) : 0.1
P(X = 5) (a single point) : 0
P(2 < X < 5)       : 0.3
Mean               : 5.0
Variance           : 8.3333
Simulated mean     : (close to 5)

Normal(0, 1)
P(-1 < Z < 1): 0.6827
```

---

## Quick Revision Points

- Range = max − min. Population variance divides by n, sample variance divides by n − 1.
- Box plot outliers lie outside Q1 − 1.5·IQR and Q3 + 1.5·IQR; whiskers stop at the last values inside the fences.
- Normal distribution: bell-shaped, symmetric around the mean; z = (x − μ) / σ; 68-95-99.7 rule.
- Discrete variable uses a PMF (probabilities add up to 1); continuous variable uses a PDF (area under the curve is 1, and P(X = x) = 0).
- Exact simulated values change with the random seed; the theoretical values (E[X] = 3.5 for a die, 5 for Uniform(0, 10)) do not.