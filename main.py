
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# === Load and preprocess dataset ===
df = pd.read_csv("./ev_battery_charging_data.csv")
df.drop(columns=['Charging Duration (min)'], inplace=True)

categorical_cols = ['Charging Mode', 'Battery Type', 'EV Model']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Optimal Charging Duration Class'])
y = df['Optimal Charging Duration Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# === Model 1: Baseline Random Forest ===
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
rf_base_preds = rf_base.predict(X_test)

# === Model 2: Logistic Regression ===
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# === Model 3: Optimized Random Forest with GridSearchCV ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train, y_train)
rf_optimized = grid_rf.best_estimator_
rf_opt_preds = rf_optimized.predict(X_test)

# === Cross-validation scores ===
cv_scores = cross_val_score(rf_optimized, X_scaled, y, cv=5, scoring='f1_weighted')
print("\nCross-validation F1 Scores (Optimized RF):", cv_scores)
print("Mean CV F1 Score:", np.mean(cv_scores))

# === Evaluation Function ===
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score (weighted):", f1_score(y_true, y_pred, average='weighted'))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plot_path = f"plots/{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to: {plot_path}")
    plt.close()

# === Evaluate All Models ===
evaluate_model(y_test, rf_base_preds, "Random Forest (Baseline)")
evaluate_model(y_test, lr_preds, "Logistic Regression")
evaluate_model(y_test, rf_opt_preds, "Random Forest (Optimized)")

# === Feature Importance from Optimized RF ===
importances = rf_optimized.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='mako')
plt.title("Feature Importances - Optimized Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("plots/feature_importance_optimized_rf.png")
print("Feature importance plot saved to: plots/feature_importance_optimized_rf.png")
plt.close()
