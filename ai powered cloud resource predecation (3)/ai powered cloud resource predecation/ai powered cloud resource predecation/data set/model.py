import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset load
data = pd.read_csv("cloud.usage.csv")

# Input features
X = data[["memory_usage", "disk_usage", "network_usage"]]

# Target
y = data["cpu_usage"]

# Training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Accuracy
score = model.score(X_test, y_test)
print("Model trained successfully!")
print("Model R² score:", score)
