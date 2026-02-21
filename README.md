# Brain Tumor Classification using 4 Machine Learning Methods

A comprehensive machine learning project implementing **four different classification algorithms** to detect and classify brain tumors from MRI images.

## 📋 Project Overview

This repository contains implementations of four distinct machine learning approaches for brain tumor classification:

1. **Convolutional Neural Network (CNN)** - Deep learning approach using neural networks
2. **Support Vector Machine (SVM)** - Statistical learning method for classification
3. **Random Forest (RF)** - Ensemble learning technique
4. **Naive Bayes (NB)** - Probabilistic classification algorithm

Each method is implemented as an independent module to compare performance, accuracy, and computational efficiency.

## 🏗️ Project Structure

```
Brain-Tumor-Classification-4ways/
├── cnn/                          # Convolutional Neural Network implementation
├── svm/                          # Support Vector Machine implementation
├── rf/                           # Random Forest implementation
├── bayes/                        # Naive Bayes implementation
├── BRISC-SwinHAFNet.pdf         # Research paper reference
└── README.md                     # This file
```

## 🎯 Objectives

- Compare the effectiveness of different machine learning algorithms on medical image classification
- Implement both traditional ML methods and deep learning approaches
- Evaluate performance metrics across all models
- Provide a framework for binary/multi-class brain tumor classification

## 📊 Dataset

The project uses MRI brain scan images for tumor classification. The data is organized and processed for each algorithm's specific requirements.

**Classes:** Brain tumor images vs. non-tumor images (or specific tumor type classifications)

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - For CNN implementation
- **Scikit-learn** - For SVM, Random Forest, and Naive Bayes
- **NumPy & Pandas** - Data manipulation and analysis
- **OpenCV** - Image processing
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development and documentation

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras scikit-learn numpy pandas opencv-python matplotlib seaborn jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DTan1101/Brain-Tumor-Classification-4ways.git
cd Brain-Tumor-Classification-4ways
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download or prepare your dataset

### Running the Models

Each folder contains Jupyter notebooks for its respective algorithm:

```bash
# For CNN
jupyter notebook cnn/brain_tumor_cnn.ipynb

# For SVM
jupyter notebook svm/brain_tumor_svm.ipynb

# For Random Forest
jupyter notebook rf/brain_tumor_rf.ipynb

# For Naive Bayes
jupyter notebook bayes/brain_tumor_bayes.ipynb
```

## 📈 Results & Comparison

Each model's performance will be evaluated based on:

- **Accuracy** - Overall correctness of predictions
- **Precision** - True positive rate among positive predictions
- **Recall** - True positive rate among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **AUC-ROC** - Area under the receiver operating characteristic curve
- **Training Time** - Computational efficiency

## 🔍 Model Descriptions

### CNN (Convolutional Neural Network)
- Deep learning approach utilizing convolutional layers for feature extraction
- Best for capturing spatial hierarchies in images
- Typically achieves high accuracy on medical image classification

### SVM (Support Vector Machine)
- Finds optimal hyperplane for classification
- Effective with high-dimensional data
- Good for binary and multi-class problems

### Random Forest
- Ensemble method combining multiple decision trees
- Robust to overfitting
- Provides feature importance rankings

### Naive Bayes
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast and interpretable

## 📚 References

- Research paper included: `BRISC-SwinHAFNet.pdf`
- Medical image classification best practices
- Scikit-learn and TensorFlow documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit issues and bug reports
- Propose improvements or new features
- Add additional classification methods
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## ✉️ Contact

For questions or inquiries, please open an issue on this repository or contact the project maintainer.

---

**Note:** This project is for educational and research purposes. Always validate results with domain experts before any medical application.