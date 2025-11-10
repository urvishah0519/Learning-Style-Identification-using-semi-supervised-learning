import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load the data
df = pd.read_csv('data_fs1.csv')

# Define the Felder-Silverman learning style dimensions based on your data labels
df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x == 0 else 0)
df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x == 3 else 0)

# 2. Separate features and target
X = df.drop(['learning_style', 'visual_verbal', 'sensing_intuitive', 'active_reflective'], axis=1)

# The new target variables are the three FSLSM dimensions
y_visual = df['visual_verbal']
y_sensing = df['sensing_intuitive']
y_active = df['active_reflective']

# 3. Train three separate models
models = {}
for name, y_data in [('visual', y_visual), ('sensing', y_sensing), ('active', y_active)]:
    print(f"Training model for {name} dimension...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, loss_function='Logloss')
    model.fit(X_resampled, y_resampled)
    models[name] = model
    print(f"✅ Model for {name} trained.")

# 4. Save all three trained models
joblib.dump(models, 'fslsm_models.joblib')

print("All FSLSM models trained and saved successfully!")