# Heart Disease Prediction Model 🫀

A comprehensive machine learning project for predicting heart disease using multiple classification algorithms with advanced preprocessing and feature selection techniques.

## 📋 Project Overview

This project implements a robust heart disease prediction system that compares multiple machine learning algorithms to identify the best performing model. The system includes comprehensive data preprocessing, feature selection, outlier removal, and model evaluation with automatic saving of the best performing model.

### 🎯 Objective
To develop an accurate and reliable machine learning model that can predict the likelihood of heart disease in patients based on various health indicators and demographic factors.

## ✨ Key Features

- **Multiple Algorithm Comparison**: Logistic Regression, SVM, Decision Tree, and Random Forest
- **Advanced Preprocessing**: Missing value handling, outlier detection, and feature scaling
- **Intelligent Feature Selection**: Univariate feature selection using statistical tests
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, precision, recall, and cross-validation
- **Automated Model Selection**: Automatically selects and saves the best performing model
- **Rich Visualizations**: Correlation matrices, scatter plots, and performance comparisons
- **Production Ready**: Complete model package with preprocessing components

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and evaluation
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **numpy** - Numerical computing
- **joblib** - Model serialization

## 📊 Dataset

The project uses a heart disease dataset containing various health indicators such as:
- **Age** - Patient age
- **Gender** - Patient gender
- **Cholesterol** - Cholesterol levels
- **Work Type** - Type of work/occupation
- **Smoking Status** - Smoking habits
- **Heart Disease** - Target variable (Yes/No)

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas matplotlib seaborn scikit-learn numpy joblib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AbdoTarek2211/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python Heart_Disease_Prediction.py
```

## 📈 Model Performance

The system evaluates four different algorithms:

| Algorithm | Key Features |
|-----------|--------------|
| **Logistic Regression** | Linear classifier with regularization |
| **Support Vector Machine** | RBF kernel with probability estimates |
| **Decision Tree** | Entropy-based with pruning parameters |
| **Random Forest** | Ensemble method with 100 estimators |

### Evaluation Metrics
- **Accuracy** - Overall correctness
- **F1-Score** - Harmonic mean of precision and recall
- **Precision** - True positive rate
- **Recall** - Sensitivity
- **Cross-Validation Score** - 5-fold CV for robust evaluation

## 🔧 Data Processing Pipeline

1. **Data Loading & Cleaning**
   - Missing value imputation using median/mode
   - Duplicate removal
   - Data type optimization

2. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Univariate feature selection (top 10 features)
   - Correlation analysis

3. **Outlier Treatment**
   - IQR-based outlier detection and removal
   - Data distribution visualization

4. **Model Training & Evaluation**
   - Stratified train-test split (70/30)
   - Feature standardization
   - Cross-validation
   - Comprehensive metric evaluation

## 📊 Output Files

The system generates several output files:

- **`best_heart_disease_model.pkl`** - Complete model package with preprocessing components
- **`best_model_[algorithm].pkl`** - Individual best performing model
- **`feature_scaler.pkl`** - Fitted StandardScaler for feature normalization
- **`feature_selector.pkl`** - Trained feature selector

## 📈 Results & Insights

- The system automatically selects the best performing model based on F1-score
- Comprehensive comparison across multiple evaluation metrics
- Feature importance analysis for medical insights
- Cross-validation ensures model robustness and generalizability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

⭐ **If you found this project helpful, please give it a star!** ⭐
