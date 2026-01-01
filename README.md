# 🏥 Hospital Readmission Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/yourusername/hospital-readmission)

> **Explainable AI system for predicting 30-day hospital readmission risk with SHAP-based interpretability and interactive dashboard deployment**

[🚀 Live Demo](https://your-app.streamlit.app) | [📊 Project Report](Hospital_Readmission_Research_Paper.pdf) | [💼 Portfolio](https://yourportfolio.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Methodology Highlights](#methodology-highlights)
- [Dashboard Demo](#dashboard-demo)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Hospital readmissions within 30 days represent a critical challenge for healthcare systems, costing billions annually and indicating potential gaps in care quality. This project develops an **explainable AI framework** that:

✅ Predicts 30-day readmission risk with **AUC-ROC of 0.638**  
✅ Achieves **51.36% recall** (catches half of actual readmissions)  
✅ Provides **SHAP-based explanations** for every prediction  
✅ Deploys as **interactive web dashboard** for clinical use  
✅ Prioritizes **transparency and interpretability** over black-box performance  

**Dataset:** 101,766 patient encounters from Diabetes 130-US Hospitals  
**Target:** Binary classification of 30-day readmission (11.16% prevalence)

**📥 Dataset Download:** The original dataset is too large for GitHub. Download from:  
🔗 [UCI Machine Learning Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
---

## ⚡ Key Features

### 🤖 Machine Learning Pipeline
- **3 models compared:** Logistic Regression, Random Forest, XGBoost
- **Class weighting** instead of SMOTE (preserves real-world distribution)
- **Hyperparameter tuning** with GridSearchCV
- **Comprehensive evaluation:** AUC-ROC, precision, recall, F1-score, confusion matrix

### 🔍 Explainable AI (SHAP)
- **Feature importance** ranking with SHAP values
- **Individual patient explanations** (force plots, waterfall plots)
- **Dependence plots** showing feature-prediction relationships
- **Clinically interpretable** risk factors for every prediction

### 📊 Interactive Dashboard
- **File upload** for batch predictions
- **Risk stratification:** Low (<30%), Medium (30-60%), High (>60%)
- **Visualizations:** Pie charts, histograms, tables
- **Download results** as CSV for documentation

### 🛠️ Feature Engineering
- **20+ derived features** capturing clinical complexity
- Age transformations, comorbidity indicators, medication complexity
- Healthcare utilization patterns, diagnosis categorization

---

## 📁 Project Structure

```
Hospital-Readmission-Prediction/
│
├── data/                                      # Data files
│   ├── hospital_readmission_final_model.pkl
│   └── hospital_readmission_prepared_data.pkl
│
├── notebooks/                                 # Jupyter notebooks
│   ├── 01_EDA_DataCleaning.ipynb             # Exploratory analysis
│   ├── 02_FeatureEngineering.ipynb           # Feature creation
│   ├── 03_ModelTraining_Evaluation.ipynb     # Model development
│   └── 04_SHAP_Explainability.ipynb          # SHAP analysis
│
├── app.py                                     # Streamlit dashboard
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
├── .gitignore                                 # Git ignore rules
├── Hospital_Readmission_Research_Paper.pdf   # Full research paper
└── images/                                    # Screenshots for README
    ├── dashboard_demo.png
    ├── shap_summary.png
    └── model_comparison.png
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hospital-readmission-prediction.git
cd hospital-readmission-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model files**
- Place `hospital_readmission_final_model.pkl` in `data/` folder
- Place `hospital_readmission_prepared_data.pkl` in `data/` folder

---

## 💻 Usage

### Run Jupyter Notebooks
Explore the complete analysis pipeline:

```bash
jupyter notebook
```

Open notebooks in order (01 → 04) to see:
- Data cleaning and EDA
- Feature engineering process
- Model training and comparison
- SHAP explainability analysis

### Run Dashboard Locally
Launch the interactive web application:

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Dashboard Features
1. **Upload patient data** (CSV format)
2. **Generate predictions** with single click
3. **View risk stratification** (Low/Medium/High)
4. **Download results** for clinical documentation

---

## 📊 Model Performance

### Final Model: Logistic Regression (Tuned)

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **AUC-ROC** | 0.638 | Acceptable discrimination (typical for readmission models) |
| **Accuracy** | 65.91% | Overall correctness rate |
| **Precision** | 16.66% | 1 true positive per ~6 flagged patients |
| **Recall** | 51.36% | Catches half of actual readmissions |
| **F1-Score** | 0.252 | Harmonic mean of precision and recall |

### Why This Performance is Clinically Appropriate

✅ **High Recall Prioritized:** Missing a readmission (false negative) costs more than unnecessary follow-up (false positive)  
✅ **Realistic AUC:** Published readmission models achieve AUC 0.60-0.70 due to inherent unpredictability  
✅ **Decision Support:** Model flags high-risk patients for clinical review, not autonomous decision-making  
✅ **Resource Optimization:** Concentrates limited follow-up resources on top 20-30% risk patients  

### Confusion Matrix

|                | Predicted: No Readmission | Predicted: Readmission |
|----------------|---------------------------|------------------------|
| **Actual: No** | 18,281 (TN)              | 9,426 (FP)            |
| **Actual: Yes**| 1,654 (FN)               | 1,169 (TP)            |

---

## 🔬 Methodology Highlights

### Why We Didn't Use SMOTE

**Critical Decision:** We avoided SMOTE (Synthetic Minority Over-sampling Technique) despite 7.96:1 class imbalance.

**Reasons:**
1. 📉 **Overfitting:** SMOTE-trained models achieved 0.83 AUC in training but collapsed to 0.58 on real patients
2. 🎯 **Miscalibration:** Synthetic 1:1 balance distorted probability predictions
3. 🧠 **Lost Interpretability:** SHAP values reflected artificial patterns, not real clinical relationships
4. ⚖️ **Clinical Validity:** Real-world prevalence (11%) must be preserved for proper risk assessment

**Our Alternative:** Class weighting (`class_weight='balanced'`) adjusts loss function without data manipulation.

### Top 5 Risk Factors (SHAP Analysis)

1. **Previous inpatient admissions** (importance: 0.297)
2. **History of any admission** (0.094)
3. **Length of stay category** (0.078)
4. **Number of diagnoses** (0.076)
5. **Age** (0.075)

---

## 🎨 Dashboard Demo

### Upload & Predict Tab
![Dashboard Upload](https://github.com/Vindya-Siriwardhana/Hospital-Readmission-Prediction/blob/main/images/dashboard_demo.png.jpg)
*Upload patient data and generate risk predictions*

### Results Tab
![Risk Distribution](https://github.com/Vindya-Siriwardhana/Hospital-Readmission-Prediction/blob/main/images/results_tab.png.jpg)

*View risk stratification and high-risk patient list*

### SHAP Explainability
![SHAP Summary](https://github.com/Vindya-Siriwardhana/Hospital-Readmission-Prediction/tree/main/images/shap_summary.png)
*Feature importance and individual patient explanations*

---

## 🔮 Future Improvements

### Technical Enhancements
- [ ] Add real-time NHS data integration via API
- [ ] Implement continuous model retraining pipeline
- [ ] Add fairness audits across demographic subgroups
- [ ] Integrate NLP for discharge summary analysis

### Clinical Validation
- [ ] Prospective validation study in NHS hospitals
- [ ] Measure intervention effectiveness (does acting on predictions reduce readmissions?)
- [ ] Conduct user experience research with clinicians
- [ ] Evaluate cost-effectiveness of targeted follow-up

### Deployment Improvements
- [ ] Dockerize application for easier deployment
- [ ] Add user authentication and role-based access
- [ ] Create mobile-responsive design
- [ ] Implement audit logging for clinical decisions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Vindya Siriwardhana**  
Data Scientist | MSc Data Science (University of Essex) | MSc Applied Statistics (University of Colombo)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/yourprofile)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:your.email@example.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- **Dataset:** Diabetes 130-US Hospitals dataset from UCI Machine Learning Repository
- **SHAP Library:** Lundberg & Lee (2017) for explainability framework
- **Streamlit:** For rapid dashboard prototyping and deployment
- **NHS Healthcare Insights:** Clinical validation from [Co-author Name], NHS Trust Hospital

---

## 📚 Research Paper

For detailed methodology, literature review, and discussion of clinical implications, see:

📄 [**Full Research Paper (PDF)**](Hospital_Readmission_Research_Paper.pdf)

**Key Topics:**
- Explainable AI in healthcare
- Class imbalance handling without SMOTE
- Precision-recall trade-offs in clinical settings
- SHAP-based interpretability for clinical trust

---

## ⭐ Show Your Support

If you find this project useful, please consider:
- ⭐ **Starring** the repository
- 🍴 **Forking** for your own experiments
- 📢 **Sharing** with colleagues in healthcare analytics

---

<div align="center">

**Built with ❤️ for improving patient care through responsible AI**

[⬆ Back to Top](#-hospital-readmission-prediction-system)

</div>
