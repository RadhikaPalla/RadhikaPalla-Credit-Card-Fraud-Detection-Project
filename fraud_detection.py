import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("creditcard.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Class Distribution
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Visualization 1: Fraud vs Normal Transactions
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# Visualization 2: Transaction Amount Distribution
plt.hist(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Confusion Matrix Visualization
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Conclusion
print("\nPROJECT COMPLETED SUCCESSFULLY")
print("Credit Card Fraud Detection performed using Random Forest.")