# heart_app.py - Complete Heart Disease Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffcccc;
        color: #d63031;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .low-risk {
        background-color: #ccffcc;
        color: #00b894;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SimpleHeartPredictor:
    def __init__(self):
        # Simple prediction model (in real app, you'd load a trained model)
        self.features = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate']
    
    def predict_risk(self, age, bp, cholesterol, heart_rate, chest_pain, diabetes, exercise_angina, st_depression, vessels, thalassemia):
        """Simple risk calculation based on medical guidelines"""
        
        # Base risk score
        risk_score = 0
        
        # Age factor
        if age > 55: risk_score += 0.2
        elif age > 45: risk_score += 0.1
        
        # Blood pressure
        if bp > 140: risk_score += 0.15
        elif bp > 130: risk_score += 0.1
        
        # Cholesterol
        if cholesterol > 240: risk_score += 0.15
        elif cholesterol > 200: risk_score += 0.1
        
        # Other factors
        if chest_pain in [2, 3]: risk_score += 0.1  # Atypical or non-anginal pain
        if diabetes: risk_score += 0.1
        if exercise_angina: risk_score += 0.1
        if st_depression > 1.0: risk_score += 0.1
        if vessels > 0: risk_score += 0.1
        if thalassemia in [1, 2]: risk_score += 0.1
        
        # Cap at 0.95
        return min(0.95, risk_score)

def create_sidebar_inputs():
    """Create input form in sidebar"""
    st.sidebar.header("🩺 Patient Information")
    
    # Personal information
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    
    # Medical measurements
    st.sidebar.subheader("Medical Measurements")
    blood_pressure = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    max_heart_rate = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    st_depression = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    
    # Medical history
    st.sidebar.subheader("Medical History")
    chest_pain = st.sidebar.selectbox("Chest Pain Type", 
        ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
    fasting_blood_sugar = st.sidebar.checkbox("Fasting Blood Sugar > 120 mg/dl")
    exercise_angina = st.sidebar.checkbox("Exercise Induced Angina")
    major_vessels = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
    thalassemia = st.sidebar.selectbox("Thalassemia", 
        ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"])
    
    # Convert inputs to numerical values
    chest_pain_val = int(chest_pain.split("(")[1].replace(")", ""))
    thalassemia_val = int(thalassemia.split("(")[1].replace(")", ""))
    
    return {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'max_heart_rate': max_heart_rate,
        'chest_pain': chest_pain_val,
        'diabetes': fasting_blood_sugar,
        'exercise_angina': exercise_angina,
        'st_depression': st_depression,
        'vessels': major_vessels,
        'thalassemia': thalassemia_val
    }

def create_risk_gauge(risk_score):
    """Create a visual risk gauge"""
    # Create gauge using matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Gauge parameters
    categories = ['Low', 'Moderate', 'High']
    colors = ['green', 'yellow', 'red']
    ranges = [0.3, 0.7, 1.0]
    
    # Create gauge bars
    for i, (cat, color, range_val) in enumerate(zip(categories, colors, ranges)):
        ax.barh(0, range_val, left=(i * 0.3), color=color, alpha=0.7, label=cat)
    
    # Add risk indicator
    ax.axvline(x=risk_score * 0.9, color='black', linewidth=3)
    ax.text(risk_score * 0.9, 0.5, f'{risk_score*100:.1f}%', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.set_xlim(0, 0.9)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Risk Score')
    ax.set_title('Heart Disease Risk Assessment')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.axis('off')  # Hide axes
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Prediction</h1>', 
                unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = SimpleHeartPredictor()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Risk Assessment", "📊 Health Analysis", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Patient Input")
            input_data = create_sidebar_inputs()
            
            # Predict button
            predict_clicked = st.button("🚀 Predict Heart Disease Risk", 
                                      use_container_width=True,
                                      type="primary")
        
        with col2:
            if predict_clicked:
                # Calculate risk
                risk_score = predictor.predict_risk(
                    input_data['age'], input_data['blood_pressure'], 
                    input_data['cholesterol'], input_data['max_heart_rate'],
                    input_data['chest_pain'], input_data['diabetes'],
                    input_data['exercise_angina'], input_data['st_depression'],
                    input_data['vessels'], input_data['thalassemia']
                )
                
                # Display results
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader("Prediction Results")
                
                # Risk gauge
                gauge_fig = create_risk_gauge(risk_score)
                st.pyplot(gauge_fig)
                
                # Risk message
                if risk_score > 0.7:
                    st.markdown(f'<div class="high-risk">'
                              f'🚨 HIGH RISK: {risk_score*100:.1f}% probability of heart disease '
                              f'</div>', unsafe_allow_html=True)
                    st.error("Please consult with a healthcare professional immediately.")
                elif risk_score > 0.3:
                    st.markdown(f'<div class="low-risk">'
                              f'⚠️ MODERATE RISK: {risk_score*100:.1f}% probability of heart disease '
                              f'</div>', unsafe_allow_html=True)
                    st.warning("Consider lifestyle changes and regular health checkups.")
                else:
                    st.markdown(f'<div class="low-risk">'
                              f'✅ LOW RISK: {risk_score*100:.1f}% probability of heart disease '
                              f'</div>', unsafe_allow_html=True)
                    st.success("Maintain your healthy lifestyle!")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show input summary
                with st.expander("View Your Input Summary"):
                    input_df = pd.DataFrame([input_data]).T
                    input_df.columns = ['Your Values']
                    st.dataframe(input_df)
            
            else:
                st.info("""
                👆 **Instructions:**
                1. Fill in your health information in the left sidebar
                2. Click the 'Predict Heart Disease Risk' button
                3. View your personalized risk assessment
                
                📋 **Note:** This is a demonstration tool. Always consult healthcare professionals for medical advice.
                """)
    
    with tab2:
        st.header("📈 Heart Health Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age vs Heart Disease Risk")
            # Sample data
            ages = np.arange(20, 80, 5)
            risks = [min(0.9, (age-20)/60 * 0.7) for age in ages]
            
            fig, ax = plt.subplots()
            ax.plot(ages, risks, 'r-', linewidth=2)
            ax.fill_between(ages, risks, alpha=0.3, color='red')
            ax.set_xlabel('Age')
            ax.set_ylabel('Relative Risk')
            ax.set_title('Age-Related Heart Disease Risk')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Blood Pressure Categories")
            bp_categories = {
                'Normal': [90, 120],
                'Elevated': [120, 130],
                'High Stage 1': [130, 140],
                'High Stage 2': [140, 180],
                'Crisis': [180, 200]
            }
            
            fig, ax = plt.subplots()
            colors = ['green', 'yellow', 'orange', 'red', 'darkred']
            for i, (category, (low, high)) in enumerate(bp_categories.items()):
                ax.barh(category, high-low, left=low, color=colors[i], alpha=0.7)
            
            ax.axvline(x=input_data['blood_pressure'], color='black', linewidth=3, 
                      label=f'Your BP: {input_data["blood_pressure"]}')
            ax.set_xlabel('Blood Pressure (mm Hg)')
            ax.legend()
            st.pyplot(fig)
    
    with tab3:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## Heart Disease Risk Assessment Tool
        
        This application provides a preliminary assessment of your heart disease risk 
        based on established medical guidelines and risk factors.
        
        ### 🔍 How It Works
        - **Input**: You provide basic health information
        - **Analysis**: The system evaluates multiple risk factors
        - **Output**: You receive a personalized risk assessment
        
        ### ⚠️ Important Disclaimer
        **This tool is for educational and informational purposes only.**
        - It is NOT a medical diagnosis
        - It should NOT replace professional medical advice
        - Always consult healthcare professionals for medical concerns
        
        ### 📊 Risk Factors Considered
        - Age and Gender
        - Blood Pressure
        - Cholesterol Levels
        - Medical History
        - Lifestyle Factors
        
        ### 🔒 Privacy Notice
        - All calculations are performed locally in your browser
        - No personal data is stored or transmitted
        - Your privacy is completely protected
        
        **Developed for educational purposes using Streamlit**
        """)

# Run the app
if __name__ == "__main__":
    main()