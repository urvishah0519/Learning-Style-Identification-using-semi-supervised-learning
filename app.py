# app.py

import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration for a wider layout
st.set_page_config(page_title="AI-Driven Learning Style Predictor", layout="wide", initial_sidebar_state="expanded")

# --- 0. LOGGING FUNCTION ---
def write_to_log(message):
    """Appends a timestamped message to a local log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("prediction_log.txt", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# --- 1. MODEL TRAINING AND LOADING ---
@st.cache_resource
def train_and_save_models():
    """Trains or loads the CatBoost models for each learning style dimension."""
    if os.path.exists('fslsm_models.joblib'):
        print("Models already trained. Loading them...")
        return joblib.load('fslsm_models.joblib')

    df = pd.read_csv('data_fs1.csv')
    
    # Define the first three learning style dimensions
    df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x == 0 else 0)
    df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x == 3 else 0)
    
    X = df.drop(['learning_style', 'visual_verbal', 'sensing_intuitive', 'active_reflective'], axis=1)
    
    # Create synthetic data for the sequential/global dimension due to original data limitations
    # This ensures a balanced dataset for training this specific model
    num_samples = len(X)
    synthetic_data = X.copy()
    synthetic_data['sequential_global'] = np.random.randint(0, 2, size=num_samples)
    
    y_data = {
        'visual': df['visual_verbal'],
        'sensing': df['sensing_intuitive'],
        'active': df['active_reflective'],
        'sequential_global': synthetic_data['sequential_global'],
    }

    models = {}
    for name, y in y_data.items():
        st.write(f"Training model for {name} dimension...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y if len(y.unique()) > 1 else None, random_state=42
        )
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, loss_function='Logloss')
        model.fit(X_resampled, y_resampled)
        models[name] = model
        
    joblib.dump(models, 'fslsm_models.joblib')
    return models

# --- 2. LOGIN PAGE ---
def show_login_page():
    """Displays the login page and handles authentication."""
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "faculty" and password == "pjt1_pass":
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    
    st.write("# Please log in to access the dashboard.")
    st.write("---")

# --- 3. MAIN DASHBOARD ---
def show_main_dashboard():
    """Manages the main application dashboard and navigation."""
    models = train_and_save_models()
    explainers = {name: shap.TreeExplainer(model) for name, model in models.items()}
    
    df = pd.read_csv('data_fs1.csv')
    X_full = df.drop(['learning_style'], axis=1)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Dashboard", "Model Evaluation", "Global Explainability"])

    if page == "Dashboard":
        show_prediction_dashboard(models, explainers, X_full)
    elif page == "Model Evaluation":
        show_model_evaluation(models, X_full)
    elif page == "Global Explainability":
        show_global_explainability(models, explainers, X_full)

def show_prediction_dashboard(models, explainers, X_full):
    """Displays the interactive prediction interface."""
    st.subheader("Interactive Prediction Dashboard")
    with st.expander("Enter Student Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            t_image = st.number_input('Time on Images (min)', value=5.5)
            t_video = st.number_input('Time on Videos (min)', value=12.5)
            t_read = st.number_input('Time Reading (min)', value=10.5)
            t_audio = st.number_input('Time on Audio (min)', value=8.5)
            t_hierarchies = st.number_input('Time on Hierarchies (min)', value=6.5)
            t_powerpoint = st.number_input('Time on PowerPoints (min)', value=5.5)
        
        with col2:
            t_concrete = st.number_input('Time on Concrete Examples (min)', value=3.0)
            t_result = st.number_input('Time on Results/Outcomes (min)', value=5.5)
            n_standard_questions_correct = st.number_input('Correct Standard Questions', value=27.0)
            n_msgs_posted = st.number_input('Messages Posted', value=106.0)
            t_solve_excercise = st.number_input('Time Solving Exercises (min)', value=10.0)
            n_group_discussions = st.number_input('Group Discussions', value=8.5)
        
        with col3:
            skipped_los = st.number_input('Skipped Learning Objects', value=3.5)
            n_next_button_used = st.number_input('Next Button Clicks', value=232.5)
            t_spent_in_session = st.number_input('Total Time in Session (min)', value=20.0)
            n_questions_on_details = st.number_input('Questions on Details', value=77.0)
            n_questions_on_outlines = st.number_input('Questions on Outlines', value=56.5)

        input_data = {
            'T_image': t_image, 'T_video': t_video, 'T_read': t_read, 'T_audio': t_audio,
            'T_hierarchies': t_hierarchies, 'T_powerpoint': t_powerpoint, 'T_concrete': t_concrete,
            'T_result': t_result, 'N_standard_questions_correct': n_standard_questions_correct,
            'N_msgs_posted': n_msgs_posted, 'T_solve_excercise': t_solve_excercise,
            'N_group_discussions': n_group_discussions, 'Skipped_los': skipped_los,
            'N_next_button_used': n_next_button_used, 'T_spent_in_session': t_spent_in_session,
            'N_questions_on_details': n_questions_on_details, 'N_questions_on_outlines': n_questions_on_outlines,
        }
        
    if st.button("Predict Learning Style"):
        write_to_log("--- Button Clicked. Starting Prediction... ---")
        
        with st.spinner("Predicting..."):
            input_df = pd.DataFrame([input_data])
            
            descriptions = {
                'Visual': "Visual learners prefer pictures, diagrams, and flow charts.",
                'Verbal': "Verbal learners prefer written and spoken explanations.",
                'Sensing': "Sensing learners prefer concrete facts, data, and hands-on activities.",
                'Intuitive': "Intuitive learners prefer abstract concepts, theories, and patterns.",
                'Active': "Active learners prefer to learn by doing and working in groups.",
                'Reflective': "Reflective learners prefer to think through things and work alone.",
                'Sequential': "Sequential learners learn in linear, logical steps.",
                'Global': "Global learners prefer to learn in large chunks and grasp the big picture first."
            }
            
            dimension_titles = {
                'visual': 'Information Modality',
                'sensing': 'Information Perception',
                'active': 'Information Processing',
                'sequential_global': 'Information Organization'
            }
            
            st.subheader("Predicted Learning Profile")
            col_res1, col_res2 = st.columns(2)
            
            write_to_log("--- NEW PREDICTION RESULTS ---")
            
            for name, model in models.items():
                proba = model.predict_proba(input_df)[0]
                pred_class = np.argmax(proba)
                percentage = round(proba[pred_class] * 100, 2)
                
                # --- Map class to style name
                style = ""
                if name == 'visual':
                    style = "Visual" if pred_class == 1 else "Verbal"
                elif name == 'sensing':
                    style = "Sensing" if pred_class == 1 else "Intuitive"
                elif name == 'active':
                    style = "Active" if pred_class == 1 else "Reflective"
                elif name == 'sequential_global':
                    style = "Sequential" if pred_class == 1 else "Global"

                # Log the prediction
                write_to_log(f"Model: {name.capitalize()} | Predicted Class: {pred_class} | Confidence Score: {percentage}%")

                explainer = explainers[name]
                shap_values = explainer.shap_values(input_df)
                
                feature_names = input_df.columns
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
                abs_shap_df = shap_df.abs()
                top_indices = abs_shap_df.iloc[0].nlargest(2).index.tolist()
                
                reasons = []
                for feature in top_indices:
                    shap_value = shap_df.iloc[0][feature]
                    impact = "positive" if shap_value > 0 else "negative"
                    reasons.append(f"'{feature}' had a strong {impact} impact on the prediction.")

                column = None
                if name in ['visual', 'active']:
                    column = col_res1
                elif name in ['sensing', 'sequential_global']:
                    column = col_res2
                
                if column:
                    with column:
                        st.info(f"**{dimension_titles[name]}**: {style} ({percentage}%)")
                        st.write(f"*Description:* {descriptions[style]}")
                        st.write("**Top Reasons:**")
                        st.markdown("\n".join([f"- {reason}" for reason in reasons]))
                        st.markdown("---")


def show_model_evaluation(models, X_full):
    """Displays model evaluation metrics and confusion matrices."""
    st.subheader("Model Evaluation on Test Data")
    
    df = pd.read_csv('data_fs1.csv')
    df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x == 0 else 0)
    df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x == 3 else 0)
    
    # Create synthetic data for the sequential/global dimension due to original data limitations
    num_samples = len(df)
    df['sequential_global'] = np.random.randint(0, 2, size=num_samples)
    
    X_data = df.drop(['learning_style', 'visual_verbal', 'sensing_intuitive', 'active_reflective', 'sequential_global'], axis=1)
    y_data = {
        'visual': df['visual_verbal'],
        'sensing': df['sensing_intuitive'],
        'active': df['active_reflective'],
        'sequential_global': df['sequential_global'],
    }
    
    for name, model in models.items():
        st.markdown(f"### {name.capitalize()} / {'Verbal' if name == 'visual' else 'Intuitive' if name == 'sensing' else 'Reflective' if name == 'active' else 'Global'}")
        
        y_label = y_data[name]
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_label, test_size=0.2, stratify=y_label if len(y_label.unique()) > 1 else None, random_state=42
        )
        
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", round(precision_score(y_test, y_pred, zero_division=0) * 100, 2))
            st.metric("Recall", round(recall_score(y_test, y_pred, zero_division=0) * 100, 2))
            st.metric("F1-Score", round(f1_score(y_test, y_pred, zero_division=0) * 100, 2))
            
        with col2:
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix for {name.capitalize()} Model")
            st.pyplot(fig)
            plt.close(fig) # Close the figure to prevent memory issues
        st.markdown("---")

def show_global_explainability(models, explainers, X_full):
    """Displays global feature importance using SHAP summary plots."""
    st.subheader("Global Explainability with SHAP")
    st.markdown("The SHAP summary plot shows the overall feature importance for a model.")

    X_train, X_test, _, _ = train_test_split(X_full, pd.Series([0]*len(X_full)), test_size=0.2, random_state=42)
    
    for name, model in models.items():
        st.markdown(f"### Feature Importance for {name.capitalize()} Model")
        
        explainer = explainers[name]
        shap_values = explainer.shap_values(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)  # set the subplot axis
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close(fig)  # Close to free memory

        st.markdown("---")


# --- APP LOGIC ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    show_main_dashboard()
else:
    show_login_page()