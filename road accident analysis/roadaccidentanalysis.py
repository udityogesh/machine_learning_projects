import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\yoges\OneDrive\Desktop\india_road_accident_dataset.csv")

print(df.head())

le = LabelEncoder()

df["State"] = le.fit_transform(df["State"])
df["Area_Type"] = le.fit_transform(df["Area_Type"])
df["Cause_of_Accident"] = le.fit_transform(df["Cause_of_Accident"])
df["Accident_Severity"] = le.fit_transform(df["Accident_Severity"])

X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

severity_counts = df["Accident_Severity"].value_counts()

plt.figure()
plt.bar(severity_counts.index, severity_counts.values)
plt.xlabel("Severity")
plt.ylabel("Count")
plt.title("Accident Severity Distribution")
plt.show()

cause_counts = df["Cause_of_Accident"].value_counts()

plt.figure()
plt.bar(cause_counts.index, cause_counts.values)
plt.xlabel("Cause")
plt.ylabel("Count")
plt.title("Cause of Accidents")
plt.show()
