import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load generated data
data = pd.read_csv('data/dataset.csv')

X = data[['avg_balance', 'monthly_inflows', 'monthly_outflows']]
y = data['risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# Save model
joblib.dump(model, 'data/model/xgb_credit_risk_model.joblib')
print("Model saved to: data/model/xgb_credit_risk_model.joblib")
