# 🧠 Brain Tumor Classification using Random Forest on BRISC2025 Dataset
## 1. Overview
This notebook implements a **Random Forest-based model** for the task of **brain tumor classification** using the **[BRISC2025 dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025)**.  
The study is part of a collaborative project where each team member develops a different model for comparison.  
This implementation serves as the **baseline traditional machine learning approach**.

---

## 2. Dataset BRISC2025 

The **BRISC2025 (Brain Tumor Segmentation and Classification)** dataset is a high-quality, expert-annotated MRI dataset introduced by *Fateh et al., 2025*.  
It addresses several issues in earlier datasets such as BraTS and Figshare, including class imbalance, limited tumor diversity, and inconsistent annotations.

- **Total samples:** 6,000 MRI images  
- **Split:** 5,000 training / 1,000 testing  
- **Modality:** T1-weighted contrast-enhanced MRI  
- **Classes:**
  - Glioma  
  - Meningioma  
  - Pituitary Tumor  
  - No Tumor  
- **Anatomical planes:** Axial, Coronal, Sagittal (balanced distribution)  


---

## 3. Pipeline


### 3.1. Data Loading & Exploration
- Load MRI images.  
- Each image represents a T1-weighted MRI slice categorized into one of four tumor classes: *glioma, meningioma, pituitary,* and *no_tumor*.  
- Perform initial exploration to understand dataset balance and visualize representative MRI slices from each category and anatomical plane.


### 3.2. Preprocessing 
- **Resize** all images to a consistent spatial resolution (128×128).  
- **Normalize** pixel intensity values to a range of `[0, 1]`.  
- **Encode** class labels numerically and split data into training and testing sets.


### 3.3. Feature Representation
- Apply **Histogram of Oriented Gradients (HOG)** to extract gradient- and texture-based features from MRI slices.  
  HOG captures local edge orientations and structural patterns that are highly relevant for tumor detection.  
- Use **Principal Component Analysis (PCA)** to reduce the high-dimensional HOG feature space into a compact representation,  
  improving computational efficiency and minimizing redundant information.


### 3.4. Model Training
- Train a **Random Forest Classifier**.  
  The model aggregates multiple decision trees to improve robustness and reduce variance.  
- Tune hyperparameters to optimize model performance:  
  - `n_estimators` (number of trees)  
  - `max_depth` (maximum depth of each tree)  
  - `max_features` (number of features considered at each split)  
  - `min_samples_split` (minimum samples required to split a node)  
  - `min_samples_leaf` (minimum samples required at a leaf node)  


---

## 4. Results Summary
| Metric | Value |
|:-------:|:------:|
| Accuracy | 0.92 |
| F1-score | 0.92 |
| Precision  | 0.92 |
| Recall  | 0.92 |

---

## 5. Conclusion:
- Random Forest achieved strong baseline performance with minimal preprocessing.  
- The model generalizes well across four tumor classes thanks to the balanced dataset.  
- However, the model shows lower accuracy for glioma and meningioma classes compared to pituitary and no_tumor. This is likely due to the morphological similarity and overlapping texture features between glioma and meningioma.


