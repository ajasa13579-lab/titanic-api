import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import urllib.request
import os

# 1. Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

data_path = 'data/titanic.csv'

# If missing, create the mock CSV for demo purposes
if not os.path.exists(data_path):
    print("Creating mock Titanic dataset inside data/titanic.csv...")
    mock_data = pd.DataFrame({
        'Pclass': [3, 1, 3, 1, 3, 2, 1, 3, 2],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0],
        'Fare': [7.2500, 71.2833, 7.9250, 53.1000, 8.0500, 8.4583, 51.8625, 21.0750, 11.1333],
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1]
    })
    mock_data.to_csv(data_path, index=False)

# 2. Setup the models output folder
output_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Load & Preprocess Data (The "Engine" cleans and fills in missing ages)
print("Loading data...")
df = pd.read_csv(data_path)

features = ['Pclass', 'Sex', 'Age', 'Fare']
X = df[features].copy()
y = df['Survived']

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 4. Train the Model ("The Brain")
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_imputed, y)

# 5. Save Model and Imputer
print("Saving the 'Brain' to models/titanic_model.joblib...")
joblib.dump(model, os.path.join(output_dir, 'titanic_model.joblib'))
joblib.dump(imputer, os.path.join(output_dir, 'titanic_imputer.joblib'))

print("All done! Training finished successfully.")
