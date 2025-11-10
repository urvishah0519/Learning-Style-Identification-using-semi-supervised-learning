from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import shap
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to allow the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained FSLSM models
try:
    models = joblib.load('fslsm_models.joblib')
    explainers = {name: shap.TreeExplainer(model) for name, model in models.items()}
    print("✅ FSLSM models and SHAP explainers loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'fslsm_models.joblib' not found. Please run 'train_model.py' first.")
    models = None
    explainers = None

# Define the data structure for your input
class StudentData(BaseModel):
    T_image: float
    T_video: float
    T_read: float
    T_audio: float
    T_hierarchies: float
    T_powerpoint: float
    T_concrete: float
    T_result: float
    N_standard_questions_correct: float
    N_msgs_posted: float
    T_solve_excercise: float
    N_group_discussions: float
    Skipped_los: float
    N_next_button_used: float
    T_spent_in_session: float
    N_questions_on_details: float
    N_questions_on_outlines: float

# Helper function to get top reasons from SHAP values
def get_top_reasons(shap_values, feature_names, top_n=2):
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    abs_shap_df = shap_df.abs()
    top_indices = abs_shap_df.iloc[0].nlargest(top_n).index.tolist()
    
    reasons = []
    for feature in top_indices:
        shap_value = shap_df.iloc[0][feature]
        impact = "positive" if shap_value > 0 else "negative"
        reasons.append(f"'{feature}' had a strong {impact} impact on the prediction.")
    return reasons


# Define the prediction endpoint
@app.post("/predict")
def predict_learning_style(data: StudentData):
    if models is None or explainers is None:
        return {"error": "Models not loaded. Please train the models and restart the server."}
    
    input_df = pd.DataFrame([data.dict()])
    
    # Initialize the output dictionary with all four dimensions and default values
    final_output = {
        "visual_verbal": {
            "style": "Verbal", "percentage": 50, "reasons": ["Prediction based on statistical average."]
        },
        "sensing_intuitive": {
            "style": "Intuitive", "percentage": 50, "reasons": ["Prediction based on statistical average."]
        },
        "active_reflective": {
            "style": "Reflective", "percentage": 50, "reasons": ["Prediction based on statistical average."]
        },
        "sequential_global": {
            "style": "Sequential", "percentage": 50, "reasons": ["Model not available for this dimension."]
        },
    }
    
    # Generate predictions and SHAP explanations for each trained model
    for name, model in models.items():
        proba = model.predict_proba(input_df)[0]
        
        explainer = explainers[name]
        shap_values = explainer.shap_values(input_df)
        
        pred_class = np.argmax(proba)

        if name == 'visual':
            style = "Visual" if pred_class == 1 else "Verbal"
            percentage = round(proba[pred_class] * 100, 2)
            reasons = get_top_reasons(shap_values, input_df.columns)
            final_output["visual_verbal"] = {"style": style, "percentage": percentage, "reasons": reasons}

        elif name == 'sensing':
            style = "Sensing" if pred_class == 1 else "Intuitive"
            percentage = round(proba[pred_class] * 100, 2)
            reasons = get_top_reasons(shap_values, input_df.columns)
            final_output["sensing_intuitive"] = {"style": style, "percentage": percentage, "reasons": reasons}
            
        elif name == 'active':
            style = "Active" if pred_class == 1 else "Reflective"
            percentage = round(proba[pred_class] * 100, 2)
            reasons = get_top_reasons(shap_values, input_df.columns)
            final_output["active_reflective"] = {"style": style, "percentage": percentage, "reasons": reasons}
        
    return final_output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)