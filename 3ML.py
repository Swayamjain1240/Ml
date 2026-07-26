import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Predict a new flower
new_flower = pd.DataFrame([{
    "sepal length (cm)": 5.9,
    "sepal width (cm)": 3.0,
    "petal length (cm)": 4.2,
    "petal width (cm)": 1.3
}])

prediction = model.predict(new_flower)[0]
predicted_species = iris.target_names[prediction]

print("New Flower Prediction:", predicted_species)