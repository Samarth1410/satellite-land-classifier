# 🛰️ Satellite Land Use Classifier

A Vision Transformer (ViT) fine-tuned on the EuroSAT dataset to classify satellite images into 10 land-use categories. Achieved **99.06% test accuracy** using transfer learning.

---

## 🎯 Overview

This project fine-tunes a pre-trained `google/vit-base-patch16-224` model on 27,000 labeled satellite images from the EuroSAT dataset. The model can classify aerial imagery into 10 land-use categories, with real-world applications in urban planning, environmental monitoring, and climate research.

---

## 📊 Results

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | 98.93% | 99.06% |
| F1 Score (macro) | 98.88% | 99.05% |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Annual Crop | 0.99 | 0.99 | 0.99 |
| Forest | 1.00 | 0.99 | 0.99 |
| Herbaceous Vegetation | 0.98 | 0.99 | 0.99 |
| Highway | 0.98 | 1.00 | 0.99 |
| Industrial Buildings | 1.00 | 0.99 | 0.99 |
| Pasture | 0.98 | 0.99 | 0.99 |
| Permanent Crop | 0.99 | 0.98 | 0.98 |
| Residential Buildings | 0.99 | 1.00 | 0.99 |
| River | 0.99 | 0.98 | 0.99 |
| SeaLake | 1.00 | 1.00 | 1.00 |

---

## 🗂️ Dataset — EuroSAT

- **Source:** Copernicus Sentinel-2 satellite imagery
- **Total images:** 27,000 (64×64 px, RGB)
- **Split:** 60% train / 20% validation / 20% test
- **Classes:** 10 land-use categories
- **Class balance:** Mostly balanced (1,195–1,863 images per class in training set)

---

## 🏗️ Model Architecture

- **Base model:** `google/vit-base-patch16-224` (pre-trained on ImageNet-21k)
- **Approach:** Full fine-tuning (all 85.8M parameters trainable)
- **Input size:** 224×224 px (upscaled from 64×64 using bilinear interpolation)
- **Output:** 10-class softmax classification head

**How ViT works:** The image is split into 16×16 pixel patches, which are treated like tokens — exactly like words in a sentence transformer. The model learns relationships between patches using self-attention.

---

## 🚀 Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Best checkpoint | Epoch 8 |
| Training time | ~82 minutes (T4 GPU) |

**Key decisions:**
- Used a small learning rate (`2e-5`) to preserve pre-trained ImageNet knowledge
- Tracked both accuracy and macro F1 to account for slight class imbalance
- Used `load_best_model_at_end=True` — training peaked at Epoch 8 before slight overfitting

---

## 📁 Project Structure

```
satellite-land-classifier/
│
├── notebooks/
│   ├── 00_dataset_download.ipynb   # Download and save EuroSAT dataset
│   ├── 01_exploration.ipynb        # EDA, class distribution, visualizations
│   ├── 02_training.ipynb           # Preprocessing, fine-tuning, saving model
│   └── 03_evaluation.ipynb         # Test set evaluation, confusion matrix
│
├── src/
│   ├── dataset.py                  # Dataset loading and preprocessing logic
│   ├── model.py                    # Model loading and configuration
│   └── inference.py                # Single image inference script
│
├── demo/
│   └── app.py                      # Gradio web app
│
├── results/
│   ├── class_samples.png           # Sample images per class
│   ├── class_distribution.png      # Training set class distribution
│   ├── pixel_distribution.png      # RGB pixel intensity distribution
│   ├── confusion_matrix.png        # Test set confusion matrix
│   └── training_curves.png         # Loss and accuracy over epochs
│
└── README.md
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Samarth1410/satellite-land-classifier.git
cd satellite-land-classifier
```

**2. Install dependencies**
```bash
pip install transformers datasets torch torchvision
pip install matplotlib scikit-learn gradio seaborn accelerate
```

**3. Download the dataset**
```bash
jupyter notebook notebooks/00_dataset_download.ipynb
```

**4. Run the Gradio demo**
```bash
cd demo
python app.py
```

---

## 🔍 Key Observations

- **Pasture** had 37% fewer training samples than other classes yet still achieved 98-99% precision and recall — demonstrating the strength of the pre-trained ViT backbone
- **SeaLake** achieved perfect 100% precision and recall — likely because water bodies have very distinct spectral signatures
- **Permanent Crop vs Annual Crop** were the hardest to distinguish — visually similar from satellite view
- Validation loss started rising after Epoch 2 while training loss kept dropping — a classic sign of overfitting, handled by saving the best checkpoint

---

## 🛠️ Tech Stack

- **Python 3.12**
- **PyTorch** — deep learning framework
- **HuggingFace Transformers** — ViT model and Trainer
- **HuggingFace Datasets** — EuroSAT dataset loading
- **Gradio** — interactive web demo
- **scikit-learn** — evaluation metrics
- **Matplotlib / Seaborn** — visualizations

---

## 📚 References

- [An Image is Worth 16x16 Words (ViT Paper)](https://arxiv.org/abs/2010.11929)
- [EuroSAT: A Novel Dataset and Deep Learning Benchmark](https://arxiv.org/abs/1709.00029)
- [HuggingFace ViT Documentation](https://huggingface.co/docs/transformers/model_doc/vit)

---

## 👤 Author

**Your Name**
[GitHub](https://github.com/Samarth1410)