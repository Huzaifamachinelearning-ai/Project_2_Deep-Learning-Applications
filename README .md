# Introduction to Applied AI: Weekly Project Log

**Author:** Muhammad Huzaifa

**Course:** Introduction to Applied AI | BS 7th Semester | Department of Electronics

**Project:** Project 2, Deep Learning Applications

**Stack:** PyTorch, ResNet-18 (ImageNet pretrained), spaCy, scikit-learn, EasyOCR, Kaggle Notebooks (NVIDIA T4)

Each week has a short summary here and a full write-up in its own README.

---

## Index

| Week | Topic | Dataset | Headline Result | Full Write-up |
| :--- | :--- | :--- | :--- | :--- |
| 5 | Scene classification with transfer learning | Intel Image Classification (6 classes) | **94.1%** test accuracy | [Week_5_README.md](Week_5_README.md) |
| 6 | Pediatric pneumonia detection in chest X-rays | Chest X-Ray Images (Pneumonia) | **0.917** macro-F1, **0.979** ROC-AUC | [Week_6_README.md](Week_6_README.md) |
| 7 | Text and multimodal AI | SMS Spam Collection (uciml) | **0.982** macro-F1 (char n-grams), NER F1 0.533 → 0.875 | [Week_7_README.md](Week_7_README.md) |

---

## Week 5: Scene Classification with Transfer Learning

**Goal:** Classify natural scenes (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`) with a two-stage transfer learning pipeline.

**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). 11,929 train / 2,105 validation / 3,000 test images.

**Approach**
1. **Feature extraction:** backbone frozen, new `nn.Linear(512, 6)` head, Adam at 1e-3, 3 epochs. Only 3,078 trainable parameters (0.03%).
2. **Fine-tuning:** `layer4` and `fc` unfrozen (~8.4M parameters) with differential learning rates: 1e-4 for `layer4`, 1e-3 for `fc`. 3 epochs.
3. **Test evaluation:** run once on all 3,000 test images.

| Phase | Trainable Params | Val Loss | Val Acc | Test Acc |
| :--- | :--- | :--- | :--- | :--- |
| Feature Extraction | 3,078 (0.03%) | 0.289 | 91.2% | - |
| Fine-Tuning (Layer 4) | ~8.4M (75.1%) | 0.218 | 94.4% | **94.1%** |

**Findings**
- Strongest classes: `forest` (F1 0.98) and `buildings` (F1 0.94).
- Weakest classes by recall: `glacier` (88.6%) and `mountain` (89.3%).
- The main confusion is glacier vs. mountain, since snowy ridges and icy slopes appear in both classes.

**Output:** model weights saved to `/kaggle/working/resnet18_intel.pt` (~44.7 MB).

---

## Week 6: Pediatric Pneumonia Detection in Chest Radiography

**Goal:** Build a ResNet-18 triage-support model for pediatric pneumonia, with a focus on data leakage, class imbalance, clinical thresholds and shortcut learning.

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). 5,840 radiographs from Guangzhou Women and Children's Medical Center (CC BY 4.0).

**Approach**

| Stage | What was done |
| :--- | :--- |
| Data audit | Imbalance 2.9:1 (1,341 Normal / 3,875 Pneumonia). 62 exact duplicates, all within-split. 0 near-duplicates between train and test. |
| Leakage control | `StratifiedGroupKFold(n_splits=6)` grouped by patient ID, with 0 shared patient groups between train and validation. |
| Imbalance | Inverse-frequency weighted cross-entropy: Normal = 1.935, Pneumonia = 0.674. |
| Augmentation | `RandomResizedCrop`, `RandomRotation(10)` and `ColorJitter`. Horizontal and vertical flips deliberately excluded because chest anatomy is asymmetric. |
| Training | Phase 1 trains the head only (1,026 parameters). Phase 2 unfreezes `layer4` with differential learning rates. Checkpoints selected by validation macro-F1. |
| Thresholding | Threshold lowered from 0.50 to **0.366**, chosen on validation to reach recall of at least 0.95. |
| Interpretability | Grad-CAM on `layer4` over 3 correct and 3 incorrect test cases. |

**Results (test set, evaluated once)**

| Metric | Value |
| :--- | :--- |
| Accuracy | 92.5% |
| Macro-F1 | 0.917 |
| ROC-AUC | 0.979 |
| Sensitivity at t = 0.366 | 99.0% (386 / 390) |
| Specificity at t = 0.366 | 80.3% (188 / 234) |
| PPV at test prevalence (62.5%) | 89.4% |
| PPV at screening prevalence (5%) | 20.9% |

Fine-tuning improved validation macro-F1 from 0.881 to 0.945 (+0.064).

**Key takeaways**
- **Shortcut learning confirmed:** Grad-CAM showed false positives driven by non-lung features such as the radiopaque **'R' marker**, shoulder soft tissue and humerus edges. Suggested fix: lung-field segmentation masking before classification.
- **Base-rate falloff:** at 5% prevalence, about 4 in 5 positive flags are false alarms despite 99% sensitivity.
- **Limitations:** single-center pediatric cohort (ages 1-5), not validated on adults, bedside AP units or other pathologies. Decision-support only, **not for autonomous clinical use**.

---

## Week 7: Text and Multimodal AI

**Goal:** Extend the Project 2 pipeline with text understanding (TF-IDF, classification, NER, regex extraction, OCR quality) and combine image and text evidence into a privacy-safe case record.

**Dataset:** [SMS Spam Collection (uciml)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). 5,572 messages (4,825 ham / 747 spam). Week 6 chest X-ray model reused for the image side.

**Approach**

| Stage | What was done |
| :--- | :--- |
| TF-IDF verification | Hand-checked scikit-learn's smooth IDF, `ln((1+N)/(1+df)) + 1`, with L2 normalization on a 3-document corpus. Cosine similarity of d2 and d3 = 0.544. |
| Text classification | `TfidfVectorizer` → `LogisticRegression` (balanced class weights). Word 1-2 grams (12,128 features) vs. char 2-5 grams (36,449 features). Baseline "always ham" = 0.866 accuracy. |
| NER | spaCy `en_core_web_sm` plus an Entity Ruler (7 patterns), scored by exact span and label match. Domain rules added for Pakistani hospitals, PKR amounts and dd/mm/yyyy dates. |
| Regex extraction | CNIC, mobile, MRN, SpO2, temperature and date patterns, each tested with positive and negative cases. |
| OCR quality | Character Error Rate (Levenshtein / reference length) with context-aware `fix_ocr` rules (O↔0, l/I↔1, N0→No). EasyOCR on a synthetic invoice. |
| Multimodal record | PII masked (CNIC, phone, email) before extraction. `REFER` if P(positive) ≥ 0.5 or urgent terms present, else `ROUTINE`. JSON output with disclaimer. |

**Results**

| Component | Metric | Before | After |
| :--- | :--- | :--- | :--- |
| Word n-gram classifier | Macro-F1 | - | **0.974** (spam P 0.979, R 0.933) |
| Char n-gram classifier | Macro-F1 | - | **0.982** |
| NER (provided, 16 entities) | F1 | 0.533 | **0.875** |
| NER (domain held-out, 14 entities) | F1 | 0.667 | **0.880** |
| OCR correction rules | CER | 0.062-0.143 | **0.000** |
| PII masking | Fields | - | CNIC, phone, email |

**Key takeaways**
- **Char n-grams beat word n-grams** and survive obfuscation: `W1N a FR33 pr1ze!! txt CLAIM now` scored P(spam) = 0.85, while `kal milte hain, 5 baje` scored 0.12.
- **Rules fill NER gaps:** the Entity Ruler lifted F1 from 0.533 to 0.875 on the provided set, but the domain held-out set (0.880) still has recall of 0.786, so rules need ongoing upkeep.
- **OCR result is optimistic:** CER 0.000 on the synthetic invoice reflects clean rendering. Real scans are expected at roughly 0.02-0.10.
- **Mask before extract:** PII redaction runs before any downstream step so identifiers never reach the case record.
- **Limitations:** decision support only, a qualified human decides. The example case record scores near the threshold (P = 0.55), so the `REFER` decision there leans on the urgent-term rule.

---

## Common Setup (Kaggle)

1. Create a new Kaggle notebook named `Week N - <Topic>`.
2. Add the week's dataset.
3. Settings: **Accelerator** GPU T4 x1 (optional for Week 7), **Internet** On.
4. Upload the week's `.ipynb` and run all cells.

---

## Progress Across Weeks

| Concept | Week 5 | Week 6 | Week 7 |
| :--- | :--- | :--- | :--- |
| Transfer learning (frozen then fine-tuned) | Yes | Yes | - |
| Differential learning rates | Yes | Yes | - |
| Test set evaluated once | Yes | Yes | - |
| Leakage and duplicate audit | - | Yes | - |
| Group-aware splitting | - | Yes | - |
| Class-imbalance handling | - | Yes | Yes |
| Decision thresholding and PPV | - | Yes | Yes |
| Grad-CAM shortcut audit | - | Yes | - |
| Model governance questions | - | Yes | Yes |
| TF-IDF and text classification | - | - | Yes |
| Named entity recognition | - | - | Yes |
| Regex field extraction | - | - | Yes |
| OCR and CER | - | - | Yes |
| PII masking | - | - | Yes |
| Multimodal evidence fusion | - | - | Yes |

---

## Template for a New Week

Copy this block, fill it in, and add a row to the Index table.

```markdown
```
## Week 8: Ship It, Show It, Team Up

**Goal:** Turn the Project 2 system (Week 6 image model + Week 7 text evidence) into a tested FastAPI web service, present it with a live API call, and form a team for Projects 3 and 4.

**Approach**

| Stage            | What was done                                                                                                           |
|:-----------------|:------------------------------------------------------------------------------------------------------------------------|
| API              | `app.py` with `/health`, `/predict` and `/analyze`. Model loaded once in `lifespan()`, inference run via `run_in_threadpool()`. |
| Configuration    | `POSITIVE = PNEUMONIA`, `URGENT` words set for the clinical domain, 5 MB upload limit.                                    |
| Error handling   | 415 wrong file type, 413 file too large, 400 corrupt image, 422 missing field or long note. No stack traces.             |
| Privacy          | CNIC and phone numbers masked before extraction, so no identifier appears in any response.                               |
| Tests            | `test_api.py`: 8 tests covering health, schema, parity with the direct model, 415, 413, 400, 422 and PII leak.           |
| Benchmark        | `bench.py`: 50 requests against `/predict` on a running server.                                                          |
| Presentation 2   | Five slides (`slides.html`), live API call, backup screenshots of a good call and a rejected call.                       |
| Team             | 60-second problem pitch, ranked top 3 problems, team charter and shared repository.                                      |

**Results**

| Metric                  | Value                                      |
|:------------------------|:-------------------------------------------|
| Tests passed            | 8 / 8                                      |
| Latency p50             | 251 ms                                     |
| Latency p95             | 522 ms (max 775 ms)                        |
| Hardware                | Intel 4 cores / 8 threads, 7.88 GB RAM, Windows 10, CPU-only PyTorch |
| Throughput (1 worker)   | ~3.5 requests/s                            |
| Capacity (Little's law) | 1 / 0.251 s ≈ 3.98 req/s ≈ 344,223 requests/day |

**Key takeaways**

- **Parity test:** the API returns the same probabilities as calling the model directly, so serving adds no hidden preprocessing drift.
- **Load once, serve many:** loading the model in `lifespan()` instead of inside the endpoint avoids a reload on every request.
- **Reject bad input early:** wrong type, oversized and corrupt files get clear 4xx codes instead of a 500.
- **Mask before extract:** the PII leak test fails if any part of a phone number or CNIC appears in the response.
- **Limitations:** decision support only, a qualified human decides. Same single-center pediatric scope as Week 6.

**Files:** `app.py`, `test_api.py`, `bench.py`, `sample.jpeg`, `slides.html`, `screenshot_good_call.png`, `screenshot_rejected_call.png`
