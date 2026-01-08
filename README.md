# Mammography Detection using Deep Learning

## Overview
This project implements breast cancer detection from mammography images using GoogleNet CNN architecture. The system utilizes advanced image preprocessing techniques and deep learning for accurate classification of mammographic images from the MIAS dataset.

## Features
- **Image Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization), Dilation, Erosion
- **Feature Extraction**: DWT (Discrete Wavelet Transform), GLCM (Gray Level Co-occurrence Matrix), HOG (Histogram of Oriented Gradients)
- **Deep Learning Models**: GoogleNet CNN and Custom CNN architectures
- **Comprehensive Evaluation**: Training and testing metrics with visualization

## Project Structure
```
Mammography-Detection-DeepLearning/
├── README.md                      # Project overview
├── requirements.txt               # pip install -r requirements.txt
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── preprocessing.py           # CLAHE, Dilation, Erosion
│   ├── feature_extraction.py      # DWT, GLCM, HOG
│   ├── models.py                  # GoogleNet, Custom CNN
│   └── train.py                   # Training, Evaluation
│
├── data/                          # MIAS dataset
│   ├── raw/                       # Original images
│   └── processed/                 # Preprocessed images
│
├── models/                        # Trained weights
│   └── mammography_model_best.h5
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Feature_Extraction.ipynb
│   └── 03_Model_Training.ipynb
│
└── results/                       # Output results
    ├── metrics/
    └── visualizations/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VENKAT200119/Mammography-Detection-DeepLearning.git
cd Mammography-Detection-DeepLearning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
This project uses the MIAS (Mammographic Image Analysis Society) dataset. Place the dataset in the `data/raw/` directory.

## Usage

### Training the Model
```bash
python src/train.py
```

### Preprocessing Images
```bash
python src/preprocessing.py
```

### Feature Extraction
```bash
python src/feature_extraction.py
```

## Results
Model performance metrics and visualizations are saved in the `results/` directory.

## Technologies Used
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors
- VENKAT200119

## Acknowledgments
- MIAS Database for providing the mammography dataset
- GoogleNet architecture for deep learning implementation
