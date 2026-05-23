import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\MANIDEEPA\OneDrive\Desktop\Predictive_project\StudentsPerformance.csv")

print(data.head())

# Convert categorical data
data['gender'] = data['gender'].map({'female':0, 'male':1})

# Features and target
X = data[['gender', 'math score', 'reading score']]
y = data['writing score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print(predictions[:5])

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()