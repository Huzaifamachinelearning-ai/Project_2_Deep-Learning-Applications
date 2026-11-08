# Project 2, Week 5: Scene Classification with Transfer Learning

An end-to-end deep learning project implementing image classification on natural scene categories using PyTorch, ResNet-18, and two-stage transfer learning (feature extraction followed by fine-tuning).

---

## 📌 Project Overview
- **Dataset**: [Intel Image Classification (puneet6060)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
  - **Classes (6)**: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
  - **Dataset Distribution**:
    - **Training Split**: 11,929 images (augmented with RandomResizedCrop, RandomHorizontalFlip, ColorJitter)
    - **Validation Split**: 2,105 images (clean 15% random split from seg_train)
    - **Test Split**: 3,000 images (`seg_test`)
- **Base Architecture**: Pretrained **ResNet-18** (ImageNet-1k weights)
- **Target Platform**: Kaggle Notebook (NVIDIA T4 GPU)

---

## 🔬 Training Strategy & Results

### 1. Stage 1: Feature Extraction (Frozen Backbone)
- **Trainable Parameters**: **3,078** parameters out of 11.2M total (0.0275% of network)
- **Configuration**:
  - Backbone: All feature extraction layers frozen (`requires_grad = False`)
  - Classifier Head: New linear classification layer (`nn.Linear(512, 6)`)
  - Optimizer: Adam, Learning Rate = $1 \times 10^{-3}$
  - Epochs: 3
- **Validation Accuracy**: **91.2%** (loss: 0.289)

### 2. Stage 2: Fine-Tuning (Differential Learning Rates)
- **Trainable Parameters**: **8,393,728** parameters (~8.4M parameters in `layer4` + `fc`)
- **Configuration**:
  - Unfrozen: `layer4` (the final residual block) and `fc` head
  - Differential LR Optimizer:
    - `model.layer4`: $\text{lr} = 1 \times 10^{-4}$ (gentle adjustment to preserve learned representations)
    - `model.fc`: $\text{lr} = 1 \times 10^{-3}$ (faster updates for domain classification)
  - Epochs: 3
- **Validation Accuracy**: **94.4%** (+3.2% absolute gain over feature extraction)

### 3. Stage 3: Test Evaluation (Unbiased Single Evaluation)
- **Test Accuracy**: **94.1%** (evaluated strictly once on all 3,000 test images)
- **Classification Performance Summary**:
  - **Strongest Classes**: `forest` (F1-score: 0.98), `buildings` (F1-score: 0.94)
  - **Weakest Classes by Recall**:
    - `glacier`: Recall **88.6%**
    - `mountain`: Recall **89.3%**
  - **Primary Confusion**: High bilateral confusion between `glacier` and `mountain` (snowy ridges and icy slopes co-occur in both classes).
- **Most Instructive Error**: A panoramic shot of snow-dusted crags labeled as `mountain` was predicted as `glacier` with 0.87 probability; the scene is genuinely ambiguous because both snow-capped mountain peaks and glacial ice are physically co-present in the identical landscape.

---

## 📊 Summary Metrics Table

| Phase | Trainable Params | Val Loss | Val Acc | Test Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Extraction** | 3,078 (0.03%) | 0.289 | 91.2% | - |
| **Fine-Tuning (Layer 4)** | ~8.4M (75.1%) | 0.218 | 94.4% | **94.1%** |

---

## 🔭 Week 6 Project 2 Proposal: Domain Adaptation

### **Domain: Automated Crop Foliage Disease Diagnosis from Field Photography**
- **Problem Statement**: Plant diseases inflict severe crop yield losses (up to 40% globally in staples like potato and tomato). Smallholder farmers frequently lack timely agricultural extension services and misdiagnose early-stage fungal and bacterial leaf blights.
- **Beneficiaries**: Agronomists, agricultural extension workers, and smallholder farmers requiring instant, offline-capable mobile diagnostic tools.
- **Candidate Dataset**: [PlantVillage Dataset (Emilio / Abdul Hasib)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
  - **License**: CC0: Public Domain / Open Access Research
  - **Scale**: ~54,305 labeled leaf images across 38 class categories (healthy foliage vs. specific pathology labels across 14 crop species).
- **Technical Plan**: Transfer learning leveraging ResNet-50 / EfficientNet-B0 pretrained backbones, fine-tuning higher-level feature extractors to detect localized chlorosis, necrotic lesions, and fungal sporulation patterns.

---

## 🛠️ How to Run
1. Open Kaggle and create a new notebook named `Week 5 - Transfer Learning`.
2. Add the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset.
3. In Kaggle Notebook Settings:
   - **Accelerator**: GPU T4 x1
   - **Internet**: On
4. Upload `Week_5_Transfer_Learning.ipynb` and run all cells.
5. The trained model state is saved to `/kaggle/working/resnet18_intel.pt` (~44.7 MB).
