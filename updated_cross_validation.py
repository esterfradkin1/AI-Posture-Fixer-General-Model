import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# ============================================================
# STEP 4 – LOADING DATASETS & PREDEFINED VALIDATION SPLIT
# ============================================================

train_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\train_datasets"
val_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\validation_datasets"


def load_folder_data(folder_path, dataset_type="train"):
    """Helper function to dynamically load files from a folder and tag them by dataset type"""
    all_data = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Directory {folder_path} does not exist.")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".csv"):
            file_full_path = os.path.join(folder_path, filename)

            if filename.endswith(".xlsx"):
                temp_df = pd.read_excel(file_full_path)
            else:
                temp_df = pd.read_csv(file_full_path)

            temp_df['Dataset_Split'] = dataset_type
            all_data.append(temp_df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


print("Loading training data (Train Folder)...")
train_df = load_folder_data(train_folder, "train")

print("Loading validation data (Validation Folder)...")
val_df = load_folder_data(val_folder, "val")

if train_df.empty or val_df.empty:
    raise RuntimeError("Both train and validation folders must contain valid data files!")

combined_df = pd.concat([train_df, val_df], ignore_index=True)

# 2. Preparing features and target labels
target_column = 'Label'
drop_cols = [target_column, 'Image_ID', 'Dataset_Split']

split_indices = np.where(combined_df['Dataset_Split'] == 'train', -1, 0)
pds = PredefinedSplit(test_fold=split_indices)

le = LabelEncoder()
y_combined = le.fit_transform(combined_df[target_column])

X_combined = combined_df.drop(columns=drop_cols, errors='ignore')
feature_names = list(X_combined.columns)

# 3. Data preprocessing (Imputation & Scaling) without data leakage
is_train = (combined_df['Dataset_Split'] == 'train').values

imputer = SimpleImputer(strategy="median")
imputer.fit(X_combined.iloc[is_train])
X_combined_imputed = imputer.transform(X_combined)

scaler = StandardScaler()
scaler.fit(X_combined_imputed[is_train])
X_combined_scaled = scaler.transform(X_combined_imputed)

# 4. Defining GRID SEARCH
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 17, 21, 23, 25],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=pds,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)

print(f"\n--- Starting Hyperparameter Tuning (Option A) ---")
print(f"Evaluating configurations directly on your separate Validation Dataset...")
grid_search.fit(X_combined_scaled, y_combined)

# ============================================================
# STEP 6 – FULL VISUALIZATIONS & DETAILED DATA PRINTING
# ============================================================

# Create the directory in advance to safely save the figures
os.makedirs('models', exist_ok=True)

# 1. Feature importance analysis
X_train_only = X_combined_scaled[is_train]
y_train_only = y_combined[is_train]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_only, y_train_only)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n" + "=" * 40)
print("   COMPLETE FEATURE IMPORTANCE RANKING (TRAIN ONLY)")
print("=" * 40)
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 12))
sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
plt.title('Complete Feature Importance Analysis (Train Dataset)')
plt.tight_layout()
plt.savefig('models/feature_importance.svg', format='svg')  # <--- Save as SVG
print("📸 Feature Importance plot saved as SVG in 'models' folder")
plt.show()

# 2. Printing Grid Search results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['param_n_neighbors'] = results_df['param_n_neighbors'].astype(int)

summary_results = results_df[[
    "param_n_neighbors", "param_metric", "param_weights",
    "mean_test_score", "mean_train_score"
]].rename(columns={
    "param_n_neighbors": "K", "param_metric": "Metric",
    "param_weights": "Weights", "mean_test_score": "Validation_Accuracy",
    "mean_train_score": "Train_Accuracy"
}).sort_values(by="Validation_Accuracy", ascending=False)

print("\n" + "=" * 40)
print("   DETAILED GRID SEARCH RESULTS (EVALUATED ON VAL FOLDER)")
print("=" * 40)
print(summary_results.to_string(index=False))

plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x='param_n_neighbors', y='mean_test_score', hue='param_metric', style='param_weights', markers=True)
plt.title(f"Validation Accuracy vs. K (Best Configuration: K={grid_search.best_params_['n_neighbors']})")
plt.ylabel("Accuracy on Validation Dataset")
plt.xlabel("Number of Neighbors (K)")
plt.tight_layout()
plt.savefig('models/grid_search_accuracy_vs_k.svg', format='svg')  # <--- Save as SVG
print("📸 Grid Search Accuracy plot saved as SVG in 'models' folder")
plt.show()

# 3. Generating classification report and actual confusion matrix for the Validation Set
X_val_only = X_combined_scaled[~is_train]
y_val_only = y_combined[~is_train]

print("\nCalculating actual confusion matrix (training a clean model on Train only)...")
real_val_model = KNeighborsClassifier(**grid_search.best_params_)
real_val_model.fit(X_train_only, y_train_only)

y_val_pred = real_val_model.predict(X_val_only)

print("\n" + "=" * 40)
print("   VALIDATION DATASET CLASSIFICATION REPORT (REAL)")
print("=" * 40)
print(classification_report(y_val_only, y_val_pred, target_names=le.classes_))

cm = confusion_matrix(y_val_only, y_val_pred)
cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in le.classes_], columns=[f"Predicted {c}" for c in le.classes_])

print("\n" + "=" * 40)
print("   CONFUSION MATRIX - VALIDATION DATASET")
print("=" * 40)
print(cm_df)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix: Validation Dataset Results')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()
plt.savefig('models/validation_confusion_matrix.svg', format='svg')  # <--- Save as SVG
print("📸 Confusion matrix saved as SVG in 'models' folder")
plt.show()

print("\n" + "!" * 40)
print(f"FINAL BEST VALIDATION ACCURACY: {grid_search.best_score_:.2%}")
print(f"BEST CONFIGURATION: {grid_search.best_params_}")
print("!" * 40)

# ============================================================
# STEP 7 – EXPORTING PIPELINE COMPONENTS FOR EXTERNAL SCRIPTS
# ============================================================
print("\nSaving model components for integration with external scripts...")
joblib.dump(real_val_model, 'models/best_posture_knn_model.pkl')
joblib.dump(imputer, 'models/posture_imputer.pkl')
joblib.dump(scaler, 'models/posture_scaler.pkl')
joblib.dump(le, 'models/posture_label_encoder.pkl')
print("✅ All components successfully saved in 'models/' directory!")

print("\n============ PIPELINE FINISHED SUCCESSFULLY ============")