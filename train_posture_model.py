import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# ============================================================
# STEP 4 – MODEL TRAINING & BLACK-BOX GENERATION
# ============================================================

# 1. LOAD DATA
train_df = pd.read_excel(r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                         r"\All individual Datasets - excel sheets\train_datasets\Final_Train_Posture_Dataset_combined.xlsx")
test_df = pd.read_excel(r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                        r"\All individual Datasets - excel sheets\test_datasets\Output_Sophie Fradkin video_Dataset.xlsx")

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# 2. SEPARATE FEATURES AND TARGETS
target_column = 'Label'
drop_cols = [target_column, 'Image_ID']

# Features and target for Training
X_train = train_df.drop(columns=drop_cols, errors='ignore')
y_train = train_df[target_column]

# Features and target for Testing
X_test = test_df.drop(columns=drop_cols, errors='ignore')
y_test = test_df[target_column]

# Ensure both files have the exact same features in the same order
feature_names = list(X_train.columns)
X_test = X_test[feature_names]

# 3. IMPUTATION
# We "Learn" the medians from the training file only
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test) # Apply train-medians to test data

# 4. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 6. FEATURE IMPORTANCE (ANALYSIS ONLY)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_imputed, y_train)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance Ranking ---")
print(importance_df)

# 7. KNN HYPERPARAMETER TUNING
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)

# 8. FINAL EVALUATION
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. SAVE BLACK-BOX MODEL  ✅ SAME PIPELINE AS LIVE
model_pack = {
    "imputer": imputer,
    "scaler": scaler,
    "model": best_knn,
    "feature_names": feature_names,
    "classes": list(best_knn.classes_)
}

joblib.dump(model_pack, "posture_knn_blackbox.joblib")
print("\nSaved black box to: posture_knn_blackbox.joblib")

# 10. VISUALIZATIONS
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df,
    palette='viridis'
)
plt.title('Feature Importance for Posture Classification')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=best_knn.classes_,
    yticklabels=best_knn.classes_
)
plt.title('Confusion Matrix – KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ============================================================
# K vs Accuracy Visualization
# ============================================================

results = pd.DataFrame(grid_search.cv_results_)

# Extract K and mean accuracy
results['k'] = results['param_n_neighbors'].astype(int)
results['mean_accuracy'] = results['mean_test_score']

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=results,
    x='k',
    y='mean_accuracy',
    marker='o'
)

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Accuracy vs Number of Neighbors')
plt.grid(True)
plt.tight_layout()
plt.show()
#=======================================================
results = pd.DataFrame(grid_search.cv_results_)

# Keep only the columns we need
results = results[[
    "param_n_neighbors", "param_metric", "param_weights",
    "mean_test_score", "std_test_score"
]].copy()

# Convert K to int for sorting/plotting
results["param_n_neighbors"] = results["param_n_neighbors"].astype(int)

# Plot: one curve per (metric, weights)
plt.figure(figsize=(9, 5))
for (metric, weights), grp in results.groupby(["param_metric", "param_weights"]):
    grp = grp.sort_values("param_n_neighbors")
    plt.plot(grp["param_n_neighbors"], grp["mean_test_score"], marker="o",
             label=f"{metric}, {weights}")

# Highlight best choice
best = grid_search.best_params_
plt.axvline(best["n_neighbors"], linestyle="--")
plt.title(f"CV Accuracy vs K (best: K={best['n_neighbors']}, {best['metric']}, {best['weights']})")
plt.xlabel("K (n_neighbors)")
plt.ylabel("Mean CV Accuracy")
plt.legend()
plt.tight_layout()
plt.show()