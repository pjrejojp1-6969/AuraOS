import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

csv_file = "ocular_dataset.csv"

# 1. Check if dataset exists
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found. Run the data collector script first!")
    exit()

print("Loading dataset...")
# 2. Load the data using Pandas
df = pd.read_csv(csv_file)

# Check if we have enough data
if len(df) < 50:
    print("Not enough data! Please record at least a few hundred rows.")
    exit()

# 3. Separate Features (X) and Labels (y)
# X = [ear_left, ear_right, iris_x]
X = df[['ear_left', 'ear_right', 'iris_x']]
# y = [label] (0=Neutral, 1=Left Wink, 2=Right Wink, 3=Long Blink)
y = df['label']

# 4. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows...")

# 5. Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# 6. Test the model and print out the "Judge Proof" metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the trained model for the Web OS to use later
model_filename = "ocular_ml_model.pkl"
joblib.dump(model, model_filename)
print(f"\nSUCCESS! Model saved as '{model_filename}'")
