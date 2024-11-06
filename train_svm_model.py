import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the dataset
data = pd.read_csv("expanded_career_recommendation_dataset.csv")

# Prepare the data
X = data.iloc[:, 1:-1]  # All columns except the first (User ID) and last (Career Path)
y = data.iloc[:, -1]    # Last column as the target (Career Path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the SVM model
with open("svm_career_recommendation_model.pkl", 'wb') as model_file:
    pickle.dump(svm_model, model_file)

print("Model saved as svm_career_recommendation_model.pkl")
