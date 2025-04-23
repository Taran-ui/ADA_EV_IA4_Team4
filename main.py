import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Load the dataset
df = pd.read_csv("ev_battery_charging_data.csv")
print(df.head())
print(df.info())
# Rename target column
df.rename(columns={"Optimal Charging Duration Class": "Charging_Class"}, inplace=True)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=["Charging Mode", "Battery Type", "EV Model"])

# Fill missing values if any (not necessary here, but safe)
df_encoded.fillna(df_encoded.mean(numeric_only=True), inplace=True)

# Split features and target
X = df_encoded.drop("Charging_Class", axis=1)
y = df_encoded["Charging_Class"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=1000, random_state=42)
}

# Store evaluation results
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} - Confusion Matrix')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    results[name] = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "Classification Report": report,
        "Model": model
    }

# Optimize Random Forest
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Evaluate optimized RF
y_pred_rf = best_rf.predict(X_test)
rf_cm = confusion_matrix(y_test, y_pred_rf)

# Save optimized RF confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax)
ax.set_title('Optimized Random Forest - Confusion Matrix')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("Optimized_Random_Forest_confusion_matrix.png")
plt.close()

# Feature importance plot
feature_importances = best_rf.feature_importances_
feature_names = X.columns

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x=feature_importances, y=feature_names, ax=ax)
ax.set_title("Feature Importances - Optimized Random Forest")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png")
plt.close()

print("✔ All models trained and evaluated successfully.")
