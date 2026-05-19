import streamlit as st
import pickle
import numpy as np

# 1. Load Model and Scaler
# Make sure these filenames match what you saved in your training script!
try:
    with open('student_burnout_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('student_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or Scaler file not found. Please run your training script first to generate 'student_burnout_model.pkl' and 'student_scaler.pkl'.")

st.title("🎓 Student Burnout Risk Predictor")
st.write("Enter the student's habits and AI usage metrics below to predict their Burnout Risk Level:")

# 2. Input Fields based on Student Dataset columns
col1, col2 = st.columns(2)

with col1:
    major = st.selectbox("Major Category", ['Humanities', 'Medical', 'Business', 'STEM', 'Arts'])
    year_of_study = st.selectbox("Year of Study", ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate'])
    pre_gpa = st.number_input("Pre-Semester GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    genai_hours = st.number_input("Weekly GenAI Usage (Hours)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
    use_case = st.selectbox("Primary Use Case for AI", ['Copywriting/Drafting', 'Ideation', 'Summarizing_Reading', 'Debugging/Troubleshooting', 'Direct_Answer_Generation'])
    prompt_skill = st.selectbox("Prompt Engineering Skill", ['Beginner', 'Intermediate', 'Advanced'])

with col2:
    tool_diversity = st.slider("Number of AI Tools Used Regularly", min_value=1, max_value=10, value=3)
    paid_sub = st.selectbox("Has Paid AI Subscription?", ["No", "Yes"])
    study_hours = st.number_input("Traditional Study Hours / Week", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
    ai_dependency = st.slider("Perceived AI Dependency (Scale 1-5)", min_value=1, max_value=5, value=3)
    policy = st.selectbox("Institutional AI Policy", ['Strict_Ban', 'Allowed_With_Citation', 'Actively_Encouraged'])
    anxiety = st.slider("Exam Anxiety Level (Scale 1-10)", min_value=1, max_value=10, value=5)
    post_gpa = st.number_input("Post-Semester GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    skill_retention = st.number_input("Skill Retention Score (0-100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

# 3. Convert dropdown selections to numerical mapping matching the training data
major_num = {'Humanities': 0, 'Medical': 1, 'Business': 2, 'STEM': 3, 'Arts': 4}[major]
year_num = {'Freshman': 0, 'Sophomore': 1, 'Junior': 2, 'Senior': 3, 'Graduate': 4}[year_of_study]
use_case_num = {'Copywriting/Drafting': 0, 'Ideation': 1, 'Summarizing_Reading': 2, 'Debugging/Troubleshooting': 3, 'Direct_Answer_Generation': 4}[use_case]
prompt_num = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}[prompt_skill]
paid_num = 1 if paid_sub == "Yes" else 0
policy_num = {'Strict_Ban': 0, 'Allowed_With_Citation': 1, 'Actively_Encouraged': 2}[policy]

# 4. Prepare data array (Must strictly match the 14 features sent into training X)
features = np.array([[
    major_num, year_num, pre_gpa, genai_hours, use_case_num, 
    prompt_num, tool_diversity, paid_num, study_hours, 
    ai_dependency, policy_num, anxiety, post_gpa, skill_retention
]])

# 5. Prediction Engine
if st.button("Predict Burnout Risk Level"):
    try:
        # Scale inputs first using the loaded scaler
        scaled_features = scaler.transform(features)
        result = model.predict(scaled_features)
        
        # Display human-readable classification
        risk_level = int(result[0])
        if risk_level == 0:
            st.success("🟢 **Predicted Burnout Risk: Low** (Student balance is stable.)")
        elif risk_level == 1:
            st.warning("🟡 **Predicted Burnout Risk: Medium** (Keep an eye on study habits.)")
        else:
            st.error("🔴 **Predicted Burnout Risk: High** (Critical levels of academic or AI strain.)")
            
    except NameError:
        st.error("Cannot make predictions because the model or scaler hasn't loaded successfully.")
