# iris_classification.py

"""
Iris Flower Classification
------------------------------------
This script trains a machine learning model to classify iris flowers
(Setosa, Versicolor, Virginica) based on petal and sepal measurements.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data   # features: sepal length, sepal width, petal length, petal width
y = iris.target # target: species (0: setosa, 1: versicolor, 2: virginica)

# Optional: Create a DataFrame for easy exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("First five rows of the dataset:")
print(df.head())

# 2. Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Feature scaling (optional but often improves performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7. Basic classification concept note
print("""
Basic Concept:
---------------
We used a supervised learning algorithm (Logistic Regression) which learns from labeled data
(features and species labels) to predict the species of new iris flowers.
The accuracy metric helps us understand how well the model performs.
""")

# End of script
