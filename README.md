# Brain Tumor Classification (4-Way)

This repository contains a comparative study for 4-class brain tumor classification using MRI images from the BRISC2025 dataset.

## Project Goal
Build and compare multiple approaches for the same classification task:
- No Tumor
- Glioma
- Meningioma
- Pituitary Tumor

Current methods included:
- Random Forest (traditional ML with HOG + PCA)
- SVM (traditional ML with HOG + PCA)
- CNN (deep learning, PyTorch)
- Bayes (placeholder notebook currently empty)

## Dataset
Reference dataset used in this project:
- BRISC2025 (Brain Tumor Segmentation and Classification)

Expected split used by notebooks:
- Train: 5,000 images
- Test: 1,000 images

Note:
- Dataset files are not committed to this repository.
- Update dataset paths inside notebooks before running.

## Clean Project Structure

```text
Brain-Tumor-Classification-4ways/
|-- README.md
|-- requirements.txt
|-- environment.yaml
|-- .gitignore
|-- data/
|   `-- labels/
|       |-- train_labels.csv
|       `-- test_labels.csv
|-- notebooks/
|   |-- bayes/
|   |   `-- bayes.ipynb
|   |-- cnn/
|   |   |-- cnn.ipynb
|   |   `-- dath_cnn.ipynb
|   |-- rf/
|   |   `-- rf.ipynb
|   `-- svm/
|       `-- multi_class_svm.ipynb
`-- docs/
    |-- BRISC-SwinHAFNet.pdf
    |-- final_report_nhom_1.pdf
    `-- final_slide_nhom_1.pdf
```

## Environment Setup

Choose one of the two options below.

### Option 1: Conda

```bash
conda env create -f environment.yaml
conda activate brain-tumor-4ways
```

### Option 2: pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## How To Run

1. Launch Jupyter:
```bash
jupyter lab
```

2. Open and run notebooks by method:
- Random Forest: `notebooks/rf/rf.ipynb`
- SVM: `notebooks/svm/multi_class_svm.ipynb`
- CNN: `notebooks/cnn/dath_cnn.ipynb`
- Bayes: `notebooks/bayes/bayes.ipynb` (currently empty)

3. Ensure dataset paths are updated in each notebook cell before training/evaluation.

## Notes
- Some notebooks were originally authored in Google Colab and may include Colab-specific imports.
- If you run locally, remove or skip Colab-only cells (`google.colab`, Drive mount, etc.).
- Results can vary depending on random seed, hardware, and preprocessing setup.

## Team Deliverables
- Technical paper: `docs/BRISC-SwinHAFNet.pdf`
- Final report: `docs/final_report_nhom_1.pdf`
- Final slides: `docs/final_slide_nhom_1.pdf`

## License
No explicit license is currently defined in this repository.
