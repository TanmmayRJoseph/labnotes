# Web Intelligence and Big Data

Internal practical – programs to learn.

---

## 1. Porter Stemmer, Snowball Stemmer and Regexp Stemmer

### Theory
- **Stemming** reduces a word to its root/stem by chopping affixes (e.g. *running → run*). The stem need not be a valid dictionary word.
- **Porter Stemmer** – rule-based, 5-phase suffix stripping, English only.
- **Snowball Stemmer** (Porter2) – improved Porter, supports many languages, more accurate.
- **Regexp Stemmer** – strips suffixes that match a user-defined regular expression.

### Installation
```bash
pip install nltk
```

### Code
```python
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer

words = ["running", "flies", "happily", "studies", "studying",
         "connected", "connection", "connecting", "generously", "cars"]

# 1. Porter Stemmer
porter = PorterStemmer()

# 2. Snowball Stemmer (language must be specified)
snowball = SnowballStemmer(language="english")

# 3. Regexp Stemmer: strips suffix patterns; min = minimum word length to stem
regexp = RegexpStemmer(r"ing$|s$|e$|able$|ed$|ly$|ion$", min=4)

print(f"{'Word':<15}{'Porter':<15}{'Snowball':<15}{'Regexp':<15}")
print("-" * 60)
for w in words:
    print(f"{w:<15}{porter.stem(w):<15}{snowball.stem(w):<15}{regexp.stem(w):<15}")
```

### Sample Observations
| Word | Porter | Snowball | Regexp |
|------|--------|----------|--------|
| running | run | run | runn |
| cars | car | car | car |
| connected | connect | connect | connect |

- Porter/Snowball give linguistically better stems; Regexp depends fully on your pattern.
- Snowball handles words like *generously* better than Porter.

---

## 2. Calculate the TF Score of a Document-Term Matrix

### Theory
**Term Frequency (TF)** measures how often a term occurs in a document:

```
TF(t, d) = (Number of times term t appears in document d)
           / (Total number of terms in document d)
```

The **Document-Term Matrix (DTM)** has documents as rows, terms as columns, and raw counts as cells.

### Code
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "data science is fun",
    "big data needs big storage",
    "web intelligence uses data mining"
]

# Step 1: Build Document-Term Matrix (raw counts)
cv = CountVectorizer()
dtm = cv.fit_transform(documents)

df_counts = pd.DataFrame(
    dtm.toarray(),
    columns=cv.get_feature_names_out(),
    index=[f"Doc{i+1}" for i in range(len(documents))]
)
print("Document-Term Matrix (counts):\n")
print(df_counts)

# Step 2: TF = count / total terms in that document
df_tf = df_counts.div(df_counts.sum(axis=1), axis=0).round(3)
print("\nTF Matrix:\n")
print(df_tf)
```

### Manual Version (without sklearn)
```python
from collections import Counter

documents = [
    "data science is fun",
    "big data needs big storage",
    "web intelligence uses data mining"
]

def compute_tf(doc):
    words = doc.lower().split()
    counts = Counter(words)
    total = len(words)
    return {word: count / total for word, count in counts.items()}

for i, doc in enumerate(documents, 1):
    print(f"Doc{i} TF:", compute_tf(doc))
```

### Example
For Doc2 = "big data needs big storage" (5 words):
- TF(big) = 2/5 = **0.4**
- TF(data) = 1/5 = **0.2**
- TF(needs) = TF(storage) = 1/5 = **0.2**

---

## 3. Confusion Matrix Metrics

**Given:** TP = 90, FN = 10, FP = 50, TN = 850, Total = 1000

|                     | Predicted Positive | Predicted Negative |
|---------------------|--------------------|--------------------|
| **Actual Positive** | TP = 90            | FN = 10            |
| **Actual Negative** | FP = 50            | TN = 850           |

### Formulas and Calculation

| Metric | Formula | Calculation | Result |
|--------|---------|-------------|--------|
| Accuracy | (TP + TN) / Total | (90 + 850) / 1000 | **0.94 (94%)** |
| Precision | TP / (TP + FP) | 90 / (90 + 50) = 90 / 140 | **0.6429 (64.29%)** |
| Recall (Sensitivity) | TP / (TP + FN) | 90 / (90 + 10) = 90 / 100 | **0.90 (90%)** |
| Specificity | TN / (TN + FP) | 850 / (850 + 50) = 850 / 900 | **0.9444 (94.44%)** |
| F1 Score | 2 × (P × R) / (P + R) | 2 × (0.6429 × 0.9) / (0.6429 + 0.9) | **0.75 (75%)** |

### Code
```python
TP, FN, FP, TN = 90, 10, 50, 850
total = TP + FN + FP + TN

accuracy    = (TP + TN) / total
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)            # also called Sensitivity / TPR
specificity = TN / (TN + FP)            # also called TNR
f1_score    = 2 * (precision * recall) / (precision + recall)

print(f"Total        : {total}")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1_score:.4f}")
```

### Output
```
Total        : 1000
Accuracy     : 0.9400
Precision    : 0.6429
Recall       : 0.9000
Specificity  : 0.9444
F1 Score     : 0.7500
```

### Interpretation
- The model is **94% accurate** overall, but accuracy is inflated by the large number of negatives (900 of 1000).
- **Recall is high (90%)** – it catches most real positives.
- **Precision is lower (64%)** – many false alarms (50 FPs).
- **F1 = 0.75** balances precision and recall.

---

### Quick Revision Points
- Stemming = rule-based chopping; Porter (English), Snowball (multi-language, better), Regexp (custom pattern).
- TF = term count ÷ total terms in the document.
- Accuracy = (TP+TN)/Total; Precision = TP/(TP+FP); Recall = TP/(TP+FN); Specificity = TN/(TN+FP); F1 = harmonic mean of precision and recall.