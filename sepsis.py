import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the training dataset
data = pd.read_csv('train.csv')

# Prepare features and target
X = data[['heart_rate', 'blood_pressure', 'respiratory_rate', 'temperature']]
y = data['sepsis_label']

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.2f}")
print(f"AUROC: {roc_auc:.2f}")

# Load the test dataset
test_data = pd.read_csv('test.csv')
test_data['sepsis_prediction'] = model.predict(test_data[['heart_rate', 'blood_pressure', 'respiratory_rate', 'temperature']])

# Save predictions
test_data[['patient_id', 'sepsis_prediction']].to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv!")

