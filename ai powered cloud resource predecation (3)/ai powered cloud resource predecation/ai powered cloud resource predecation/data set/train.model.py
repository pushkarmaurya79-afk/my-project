import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Data load
data = pd.read_csv("cloud.usage.csv")

# Features
X = data[["cpu_usage", "memory_usage", "disk_usage"]]

# Target
y = data["cloud_usage"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("R2 Score:", round(r2_score(y_test, pred), 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))

# Save model
joblib.dump(model, "cloud_model.pkl")

print("Model successfully trained!")
print("Features:", list(X.columns))
print("cloud_model.pkl successfully created!")\






