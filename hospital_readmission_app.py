"""
Hospital Readmission Prediction Dashboard
Author: Vindya Siriwardhana
Description: Interactive web app for predicting 30-day hospital readmissions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066CC;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333333;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0066CC;
    }
    .high-risk {
        color: #dc3545;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffc107;
        font-weight: bold;
    }
    .low-risk {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        # Updated paths
        model_path = 'C:/Users/USER/Desktop/Github Project/New folder/data/hospital_readmission_final_model.pkl'
        data_path = 'C:/Users/USER/Desktop/Github Project/New folder/data/hospital_readmission_prepared_data.pkl'
        
        with open(model_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        
        with open(data_path, 'rb') as f:
            data_artifacts = pickle.load(f)
        
        return model_artifacts, data_artifacts
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure model files are in: C:/Users/USER/Desktop/Github Project/New folder/data/")
        return None, None

# Feature engineering function
def engineer_features(df):
    """Apply same feature engineering as training"""
    
    df_fe = df.copy()
    
    # 1. Age numeric
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df_fe['age_numeric'] = df_fe['age'].map(age_mapping)
    
    # 2. Age category
    df_fe['age_category'] = pd.cut(df_fe['age_numeric'], 
                                    bins=[0, 40, 60, 80, 100], 
                                    labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # 3. Polypharmacy
    df_fe['polypharmacy'] = (df_fe['num_medications'] >= 5).astype(int)
    
    # 4. Comorbidity
    df_fe['comorbidity_count'] = df_fe['number_diagnoses']
    df_fe['high_comorbidity'] = (df_fe['number_diagnoses'] >= 7).astype(int)
    
    # 5. Length of stay category
    df_fe['los_category'] = pd.cut(df_fe['time_in_hospital'],
                                    bins=[0, 3, 7, 14],
                                    labels=['Short', 'Medium', 'Long'])
    
    # 6. Previous visits
    df_fe['had_emergency'] = (df_fe['number_emergency'] > 0).astype(int)
    df_fe['had_inpatient'] = (df_fe['number_inpatient'] > 0).astype(int)
    df_fe['had_outpatient'] = (df_fe['number_outpatient'] > 0).astype(int)
    df_fe['total_previous_visits'] = (df_fe['number_emergency'] + 
                                       df_fe['number_inpatient'] + 
                                       df_fe['number_outpatient'])
    
    # 7. Lab procedures
    df_fe['high_lab_procedures'] = (df_fe['num_lab_procedures'] > 50).astype(int)
    
    # 8. Diagnosis categories
    def categorize_diagnosis(diag):
        if pd.isna(diag) or diag == '?':
            return 'Unknown'
        diag = str(diag)
        if diag.startswith('V') or diag.startswith('E'):
            return 'Other'
        try:
            code = float(diag)
        except:
            return 'Other'
        
        if 390 <= code <= 459 or code == 785:
            return 'Circulatory'
        elif 460 <= code <= 519 or code == 786:
            return 'Respiratory'
        elif 520 <= code <= 579 or code == 787:
            return 'Digestive'
        elif 250 <= code < 251:
            return 'Diabetes'
        elif 800 <= code <= 999:
            return 'Injury'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        elif 580 <= code <= 629 or code == 788:
            return 'Genitourinary'
        elif 710 <= code <= 739:
            return 'Musculoskeletal'
        elif 780 <= code <= 799:
            return 'Symptoms'
        else:
            return 'Other'
    
    df_fe['diag_1_category'] = df_fe['diag_1'].apply(categorize_diagnosis)
    df_fe['has_circulatory'] = (df_fe['diag_1_category'] == 'Circulatory').astype(int)
    df_fe['has_respiratory'] = (df_fe['diag_1_category'] == 'Respiratory').astype(int)
    df_fe['has_diabetes_complication'] = (df_fe['diag_1_category'] == 'Diabetes').astype(int)
    
    # 9. Medication features
    df_fe['medication_changed'] = (df_fe['change'] == 'Ch').astype(int)
    df_fe['on_diabetes_med'] = (df_fe['diabetesMed'] == 'Yes').astype(int)
    
    # Count active diabetes medications
    med_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                   'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                   'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                   'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                   'glyburide-metformin', 'glipizide-metformin',
                   'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                   'metformin-pioglitazone']
    
    def count_active_meds(row):
        count = 0
        for med in med_columns:
            if med in row.index and row[med] not in ['No', 'Steady']:
                count += 1
        return count
    
    df_fe['diabetes_meds_count'] = df_fe.apply(count_active_meds, axis=1)
    
    return df_fe

def prepare_for_prediction(df, scaler, label_encoders, feature_names):
    """Prepare data for model prediction"""
    
    # Select numerical features
    feature_columns = [
        'age_numeric', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_diagnoses', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'total_previous_visits', 'polypharmacy', 'high_comorbidity',
        'high_lab_procedures', 'had_emergency', 'had_inpatient', 'had_outpatient',
        'has_circulatory', 'has_respiratory', 'has_diabetes_complication',
        'diabetes_meds_count', 'medication_changed', 'on_diabetes_med'
    ]
    
    # Categorical features
    categorical_features = ['gender', 'age_category', 'los_category', 'diag_1_category']
    
    # Encode categorical features
    for col in categorical_features:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            df[col + '_encoded'] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            df[col + '_encoded'] = 0
    
    # Combine all features
    final_features = feature_columns + [col + '_encoded' for col in categorical_features]
    X = df[final_features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=final_features)
    
    return X_scaled

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🏥 Hospital Readmission Prediction System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Predict 30-day hospital readmission risk using machine learning
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_artifacts, data_artifacts = load_model_artifacts()
    
    if model_artifacts is None or data_artifacts is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    model = model_artifacts['model']
    metrics = model_artifacts['metrics']
    scaler = data_artifacts['scaler']
    label_encoders = data_artifacts['label_encoders']
    feature_names = data_artifacts['feature_names']
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Model:** {model_artifacts['model_name']}
        
        **Performance Metrics:**
        - AUC-ROC: {metrics['auc']:.3f}
        - Recall: {metrics['recall']:.1%}
        - Precision: {metrics['precision']:.1%}
        
        **Top Risk Factors:**
        1. Previous admissions
        2. Length of stay
        3. Number of diagnoses
        """)
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool predicts the risk of hospital readmission within 30 days.
        
        **Note:** High recall (51%) prioritizes catching readmissions over false alarms - 
        appropriate for clinical decision support.
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Results", "ℹ️ How to Use"])
    
    with tab1:
        st.markdown('<div class="sub-header">Upload Patient Data</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload CSV file with patient data",
                type=['csv'],
                help="Upload a CSV file with the same format as the training data"
            )
        
        with col2:
            st.markdown("**Required columns:**")
            st.caption("age, gender, time_in_hospital, num_medications, number_inpatient, etc.")
            
            if st.button("📄 Download Sample CSV"):
                # Create sample data
                sample_data = pd.DataFrame({
                    'age': ['[70-80)'],
                    'gender': ['Female'],
                    'time_in_hospital': [5],
                    'num_lab_procedures': [45],
                    'num_procedures': [2],
                    'num_medications': [15],
                    'number_outpatient': [0],
                    'number_emergency': [0],
                    'number_inpatient': [1],
                    'number_diagnoses': [9],
                    'diag_1': ['428'],
                    'change': ['Ch'],
                    'diabetesMed': ['Yes']
                })
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="sample_patient_data.csv",
                    mime="text/csv"
                )
        
        if uploaded_file is not None:
            try:
                # Load data
                df_raw = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(df_raw)} patients loaded.")
                
                with st.expander("📋 View uploaded data"):
                    st.dataframe(df_raw.head(10))
                
                # Process button
                if st.button("🔮 Generate Predictions", type="primary"):
                    with st.spinner("Processing data and generating predictions..."):
                        # Feature engineering
                        df_engineered = engineer_features(df_raw)
                        
                        # Prepare for prediction
                        X_prepared = prepare_for_prediction(
                            df_engineered, scaler, label_encoders, feature_names
                        )
                        
                        # Make predictions
                        predictions = model.predict(X_prepared)
                        prediction_proba = model.predict_proba(X_prepared)[:, 1]
                        
                        # Add results to dataframe
                        df_results = df_raw.copy()
                        df_results['Risk_Score'] = (prediction_proba * 100).round(2)
                        df_results['Prediction'] = predictions
                        df_results['Risk_Category'] = pd.cut(
                            prediction_proba,
                            bins=[0, 0.3, 0.6, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        # Store in session state
                        st.session_state['results'] = df_results
                        st.session_state['predictions'] = predictions
                        st.session_state['probabilities'] = prediction_proba
                        
                        st.success("✅ Predictions generated successfully!")
                        st.info("👉 Go to the 'Results' tab to view predictions")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV has all required columns")
    
    with tab2:
        st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.info("👈 Please upload data and generate predictions first")
        else:
            results = st.session_state['results']
            probabilities = st.session_state['probabilities']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Patients", len(results))
            
            with col2:
                high_risk = (results['Risk_Category'] == 'High').sum()
                st.metric("High Risk", high_risk, 
                         delta=f"{high_risk/len(results)*100:.1f}%")
            
            with col3:
                medium_risk = (results['Risk_Category'] == 'Medium').sum()
                st.metric("Medium Risk", medium_risk,
                         delta=f"{medium_risk/len(results)*100:.1f}%")
            
            with col4:
                low_risk = (results['Risk_Category'] == 'Low').sum()
                st.metric("Low Risk", low_risk,
                         delta=f"{low_risk/len(results)*100:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = results['Risk_Category'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Distribution",
                    color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Risk score histogram
                fig_hist = px.histogram(
                    results,
                    x='Risk_Score',
                    nbins=20,
                    title="Risk Score Distribution",
                    labels={'Risk_Score': 'Risk Score (%)'},
                    color_discrete_sequence=['#0066CC']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # High risk patients table
            st.markdown("### 🔴 High Risk Patients (Require Immediate Attention)")
            high_risk_patients = results[results['Risk_Category'] == 'High'].sort_values(
                'Risk_Score', ascending=False
            )
            
            if len(high_risk_patients) > 0:
                display_cols = ['Risk_Score', 'Risk_Category']
                # Add available columns
                for col in ['age', 'gender', 'time_in_hospital', 'number_inpatient']:
                    if col in high_risk_patients.columns:
                        display_cols.append(col)
                
                st.dataframe(
                    high_risk_patients[display_cols].head(10),
                    use_container_width=True
                )
            else:
                st.success("No high-risk patients identified!")
            
            # Download results
            st.markdown("### 💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="readmission_predictions.csv",
                    mime="text/csv"
                )
            
            with col2:
                high_risk_csv = high_risk_patients.to_csv(index=False)
                st.download_button(
                    label="⚠️ Download High Risk List (CSV)",
                    data=high_risk_csv,
                    file_name="high_risk_patients.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.markdown('<div class="sub-header">How to Use This Tool</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📝 Step-by-Step Guide
        
        1. **Prepare Your Data**
           - Ensure your CSV file contains all required patient information
           - Download the sample CSV to see the expected format
        
        2. **Upload Data**
           - Go to the "Upload & Predict" tab
           - Click "Browse files" and select your CSV file
           - Review the uploaded data to ensure it loaded correctly
        
        3. **Generate Predictions**
           - Click the "Generate Predictions" button
           - Wait for the model to process (usually a few seconds)
        
        4. **Review Results**
           - Switch to the "Results" tab
           - View summary statistics and visualizations
           - Check the high-risk patients table
           - Download results for your records
        
        ### 🎯 Interpreting Risk Scores
        
        - **Low Risk (0-30%):** Standard discharge procedures
        - **Medium Risk (30-60%):** Consider follow-up phone call
        - **High Risk (60-100%):** Arrange post-discharge visit, medication review
        
        ### ⚠️ Important Notes
        
        - This tool is for **decision support** only, not diagnosis
        - Clinical judgment should always take precedence
        - Model has 51% recall (catches half of readmissions)
        - 16% precision (5-6 false alarms per true positive) - intentional trade-off
        
        ### 💡 Best Practices
        
        - Use risk scores to prioritize follow-up resources
        - Combine with clinical assessment
        - Document interventions for high-risk patients
        - Monitor outcomes to validate predictions
        """)

if __name__ == "__main__":
    main()
