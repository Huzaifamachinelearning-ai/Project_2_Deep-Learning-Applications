# Project 2: Deep Learning Applications — Week 7

**Course:** Introduction to Applied AI | BS 8th Semester | Department of Electronics  
**Platform:** Kaggle Notebook (GPU optional)

---

## Week 7: Text and Multimodal AI

This week extends the Project 2 pipeline with text understanding: TF-IDF feature extraction, supervised text classification, named entity recognition, structured field extraction via regex, OCR quality measurement, and a privacy-safe multimodal case record combining image and text evidence.

---

## 1. TF-IDF Verification

Scikit-learn's `TfidfVectorizer` uses **smooth IDF** with **L2 normalization**. Given a corpus of $N$ documents and a term appearing in $df$ documents:

$$\text{IDF}(t) = \ln\!\left(\frac{1 + N}{1 + df_t}\right) + 1$$

Each document vector $\mathbf{d}$ is then scaled to unit length:

$$\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2}$$

**Worked example** on corpus `['win free cash now', 'call me now', 'free call now']` with $N = 3$:

| Term | $df$ | $\text{IDF}$ | $\text{TF-IDF}_{d_1}$ (after L2) |
|------|------|-------------|----------------------------------|
| win  | 1 | $\ln(4/2) + 1 = 1.693$ | **0.584** |
| free | 2 | $\ln(4/3) + 1 = 1.288$ | **0.445** |
| cash | 1 | $\ln(4/2) + 1 = 1.693$ | **0.584** |
| now  | 3 | $\ln(4/4) + 1 = 1.000$ | **0.345** |

"now" receives the lowest weight because $df = N$, meaning it appears in every document and carries zero discriminative information ($\ln(1) = 0$).

**Cosine similarity** between $d_2$ and $d_3$:

$$\cos(d_2, d_3) = \sum_i \hat{d}_{2,i} \cdot \hat{d}_{3,i} = (0.548 \times 0.620) + (0.425 \times 0.481) = \mathbf{0.544}$$

---

## 2. Text Classification

**Dataset:** SMS Spam Collection (uciml) — 5,572 messages (4,825 ham / 747 spam).

**Baseline:** A naive "always ham" classifier achieves accuracy $= 1 - 0.134 = 0.866$, so any useful model must substantially exceed this.

### Model Comparison

| Model | Features | Macro-F1 | Spam Precision | Spam Recall |
|-------|----------|----------|----------------|-------------|
| Word 1–2 grams | 12,128 | **0.974** | 0.979 | 0.933 |
| Char 2–5 grams | 36,449 | **0.982** | — | — |

Both pipelines use `TfidfVectorizer` → `LogisticRegression` (balanced class weights, `max_iter=1000`).

The **macro-averaged F1** is defined as:

$$F_1^{\text{macro}} = \frac{1}{|C|}\sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

### Learned Features

| Direction | Top 12 Features |
|-----------|----------------|
| Most spam-like | `call`, `txt`, `free`, `text`, `reply`, `uk`, `www`, `mobile`, `claim`, `stop`, `your` |
| Most ham-like | `my`, `me`, `ok`, `that`, `it`, `but`, `ll`, `gt`, `lt`, `can`, `what`, `da` |

### Robustness to Obfuscation

| Input | Char Model $P(\text{spam})$ |
|-------|-----------------------------|
| `W1N a FR33 pr1ze!! txt CLAIM now` | **0.85** |
| `kal milte hain, 5 baje` | 0.12 |

Character n-grams detect leet-speak spam because sub-word patterns survive character substitutions.

---

## 3. Named Entity Recognition (NER)

Entities are evaluated with **exact-match** scoring — both the text span and the label must match the gold annotation:

$$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}, \quad F_1 = \frac{2PR}{P + R}$$

### Provided Sentences (6 sentences, 16 gold entities)

| Configuration | Precision | Recall | $F_1$ |
|---------------|-----------|--------|-------|
| spaCy out-of-the-box | 0.571 | 0.500 | **0.533** |
| + Entity Ruler (7 patterns) | 0.875 | 0.875 | **0.875** |

### Domain Held-Out Set (5 healthcare sentences, 14 gold entities)

| Configuration | Precision | Recall | $F_1$ |
|---------------|-----------|--------|-------|
| Base entity ruler | 0.800 | 0.571 | **0.667** |
| + Domain hospital rules | 1.000 | 0.786 | **0.880** |

Entity ruler patterns added: `MONEY` (PKR/Rs. formats), `ORG` (Engro Fertilizers, Meezan Bank, PIMS, Jazz, and 5 Pakistani hospitals), `FAC` (Karachi Port), `DATE` (dd/mm/yyyy regex).

---

## 4. Regex Field Extraction

Six base patterns and three domain-specific patterns, each validated with **positive and negative tests**:

| Pattern | Regex | Positive Match | Negative Rejection |
|---------|-------|----------------|-------------------|
| CNIC | `\b\d{5}-\d{7}-\d\b` | `35202-1234567-1` ✓ | 14-digit order number ✗ |
| Mobile | `(?:\+92[- ]?\|\\b0)3\d{2}[- ]?\d{7}\b` | `+92 300 1234567` ✓ | `Room 0300` ✗ |
| MRN | `\bMRN-\d{6}\b` | `MRN-482910` ✓ | `MRN-999` (3 digits) ✗ |
| SpO2 | `\bSpO2:\s*(?:100\|\d{2})%\b` | `SpO2: 94%` ✓ | `SpO2: 180%` ✗ |
| Temp | `\b\d{2}(?:\.\d)?\s*°C\b` | `38.5 °C` ✓ | — |

---

## 5. OCR Quality — Character Error Rate

The **Character Error Rate (CER)** measures edit distance relative to the reference length:

$$\text{CER} = \frac{\text{Levenshtein}(r, h)}{\max(|r|, 1)}$$

where $r$ is the reference string and $h$ is the OCR hypothesis.

### Correction Rules (`fix_ocr`)

Context-aware substitutions for common OCR confusions (O↔0, l/I↔1, N0→No):

| OCR Hypothesis | CER (raw) | After `fix_ocr` | CER (fixed) |
|----------------|-----------|------------------|-------------|
| `Invoice N0. 4417` | 0.062 | `Invoice No. 4417` | **0.000** |
| `Total: PKR 12,5O0` | 0.059 | `Total: PKR 12,500` | **0.000** |
| `Due 3O/09/2O26` | 0.143 | `Due 30/09/2026` | **0.000** |

### Synthetic Invoice (EasyOCR)

| Metric | Value |
|--------|-------|
| CER raw | **0.000** |
| CER after fixes | **0.000** |

The synthetic image achieves perfect OCR due to clean rendering (high contrast, standard font, no distortion). Real scanned documents are expected to yield CER in the range 0.02–0.10.

---

## 6. Multimodal Case Record Pipeline

### Architecture

```
                    ┌─────────────┐
Clinical Note ────► │ mask_pii()  │──► text_evidence() ──┐
                    └─────────────┘                       │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ case_record() │──► JSON
                                                   └──────────────┘
                                                          ▲
X-ray Image ──────────► image_evidence() ────────────────┘
```

### Decision Logic

The system outputs **REFER** if either condition holds:

$$\text{decision} = \begin{cases} \texttt{REFER} & \text{if } P(\text{positive}) \geq \tau \;\lor\; \text{urgent\_terms} \neq \emptyset \\ \texttt{ROUTINE} & \text{otherwise} \end{cases}$$

where $\tau = 0.5$ is the classification threshold.

### Privacy Masking

All PII is redacted **before** any downstream extraction to prevent data leakage:

| Field | Replacement |
|-------|-------------|
| CNIC (`35202-1234567-1`) | `[CNIC]` |
| Phone (`0300-1234567`) | `[PHONE]` |
| Email (`a@b.pk`) | `[EMAIL]` |

### Example Output

```json
{
  "image": "invoice.png",
  "image_probabilities": {
    "NORMAL": 0.452,
    "PNEUMONIA": 0.548
  },
  "note_masked": "Child aged 4, fever for 3 days and fast breath. Seen at PIMS Islamabad on 12/09/2026. Guardian mobile [PHONE].",
  "text_evidence": {
    "entities": [["4","CARDINAL"],["3 days","DATE"],["PIMS","ORG"],["Islamabad","GPE"],["12/09/2026","DATE"]],
    "fields": {"date": ["12/09/2026"]},
    "urgent_terms": ["fever","breath"]
  },
  "decision": "REFER",
  "reasons": [
    "image model: P(PNEUMONIA) = 0.55",
    "note mentions: fever, breath"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

---

## Results Summary

| Component | Metric | Before | After |
|-----------|--------|--------|-------|
| Word n-gram classifier | Macro-$F_1$ | — | **0.974** |
| Char n-gram classifier | Macro-$F_1$ | — | **0.982** |
| NER (provided) | $F_1$ | 0.533 | **0.875** |
| NER (domain held-out) | $F_1$ | 0.667 | **0.880** |
| OCR correction rules | CER | 0.062–0.143 | **0.000** |
| PII masking | Fields | — | CNIC, phone, email |

---

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| **spaCy** (`en_core_web_sm`) | Tokenization, NER, entity ruler |
| **scikit-learn** | TF-IDF vectorization, logistic regression, metrics |
| **EasyOCR** | Optical character recognition |
| **PyTorch + torchvision** | Week 6 image classification model |
| **Pillow (PIL)** | Image generation and processing |
| **NumPy / Pandas** | Numerical computation and data manipulation |
