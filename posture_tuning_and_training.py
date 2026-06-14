import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

print("============ STARTING POSTURE PIPELINE (TUNING + FIXED TRAIN+VAL TRAINING) ============")

# ============================================================
# STEP 1 – DEFINING PATHS & DATA LOADING HELPER
# ============================================================
train_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\train_datasets"
val_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\validation_datasets"
test_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\test_datasets"


def load_folder_data(folder_path, dataset_type=None):
    """
    Dynamically loads Excel or CSV files from a folder and optionally
    marks them with a 'Dataset_Split' identifier.
    """
    all_data = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: The folder {folder_path} does not exist.")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".csv"):
            file_full_path = os.path.join(folder_path, filename)

            if filename.endswith(".xlsx"):
                temp_df = pd.read_excel(file_full_path)
            else:
                temp_df = pd.read_csv(file_full_path)

            if dataset_type is not None:
                temp_df['Dataset_Split'] = dataset_type

            all_data.append(temp_df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# Load all datasets
print("Loading training data (Train Folder)...")
train_df = load_folder_data(train_folder, "train")

print("Loading validation data (Validation Folder)...")
val_df = load_folder_data(val_folder, "val")

print("Loading test data (Test Folder)...")
test_df = load_folder_data(test_folder)

if train_df.empty or val_df.empty or test_df.empty:
    raise RuntimeError("Train, Validation, and Test folders must all contain valid data files!")

# Create models directory for saving components and visualizations
os.makedirs('models', exist_ok=True)

# ============================================================
# STEP 2 – COMBINING TRAIN & VALIDATION FOR PIPELINE PREPROCESSING
# ============================================================
print(f"Merging: Train ({len(train_df)} rows) + Validation ({len(val_df)} rows) as per original configuration...")
combined_tuning_df = pd.concat([train_df, val_df], ignore_index=True)

target_column = 'Label'
drop_cols = [target_column, 'Image_ID', 'Dataset_Split']

# Fit final Label Encoder on the fully combined set
le_final = LabelEncoder()
y_tuning = le_final.fit_transform(combined_tuning_df[target_column])
y_test = le_final.transform(test_df[target_column])

X_tuning = combined_tuning_df.drop(columns=drop_cols, errors='ignore')
X_test = test_df.drop(columns=drop_cols, errors='ignore')
feature_names = list(X_tuning.columns)

# Preprocessing: Imputation and Scaling fitted on the FULL combined dataset (No subsetting)
imputer_final = SimpleImputer(strategy="median")
X_tuning_imputed = imputer_final.fit_transform(X_tuning)
X_test_imputed = imputer_final.transform(X_test)

scaler_final = StandardScaler()
X_tuning_scaled = scaler_final.fit_transform(X_tuning_imputed)
X_test_scaled = scaler_final.transform(X_test_imputed)

# ============================================================
# STEP 3 – GRID SEARCH (HYPERPARAMETER TUNING ANALYSIS)
# ============================================================
# PredefinedSplit indices mapping (-1 for train, 0 for validation)
split_indices = np.where(combined_tuning_df['Dataset_Split'] == 'train', -1, 0)
pds = PredefinedSplit(test_fold=split_indices)

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

print(f"\n--- Running Hyperparameter Tuning Grid Search ---")
grid_search.fit(X_tuning_scaled, y_tuning)

print("\n" + "!" * 40)
print(f"GRID SEARCH BEST ACCURACY: {grid_search.best_score_:.2%}")
print(f"GRID SEARCH BEST CONFIGURATION: {grid_search.best_params_}")
print("!" * 40)

# ============================================================
# STEP 4 – VALIDATION VISUALIZATIONS & ANALYSIS
# ============================================================
# 1. Feature Importance Analysis
is_train_tuning = (combined_tuning_df['Dataset_Split'] == 'train').values
X_train_only = X_tuning_scaled[is_train_tuning]
y_train_only = y_tuning[is_train_tuning]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_only, y_train_only)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n" + "=" * 50)
print("   COMPLETE FEATURE IMPORTANCE RANKING")
print("=" * 50)
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 12))
sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
plt.title('Complete Feature Importance Analysis (Train Dataset)')
plt.tight_layout()
plt.savefig('models/feature_importance.svg', format='svg')
print("📸 Feature Importance plot saved as SVG.")
plt.show()

# 2. Grid Search Results Visualization
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['param_n_neighbors'] = results_df['param_n_neighbors'].astype(int)

plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x='param_n_neighbors', y='mean_test_score', hue='param_metric', style='param_weights',
             markers=True)
plt.title("Grid Search Validation Accuracy vs. K")
plt.ylabel("Accuracy on Validation Dataset")
plt.xlabel("Number of Neighbors (K)")
plt.tight_layout()
plt.savefig('models/grid_search_accuracy_vs_k.svg', format='svg')
print("📸 Grid Search Accuracy plot saved as SVG.")
plt.show()

# 3. Evaluation on Validation Set using original model hyper-parameters
X_val_only = X_tuning_scaled[~is_train_tuning]
y_val_only = y_tuning[~is_train_tuning]

original_val_model = KNeighborsClassifier(n_neighbors=21, metric='manhattan', weights='distance', n_jobs=-1)
original_val_model.fit(X_train_only, y_train_only)
y_val_pred = original_val_model.predict(X_val_only)

print("\n" + "=" * 50)
print("   VALIDATION DATASET CLASSIFICATION REPORT (ORIGINAL PARAMS)")
print("=" * 50)
print(classification_report(y_val_only, y_val_pred, target_names=le_final.classes_))

cm_val = confusion_matrix(y_val_only, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=le_final.classes_, yticklabels=le_final.classes_)
plt.title('Confusion Matrix: Validation Dataset Results')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()
plt.savefig('models/validation_confusion_matrix.svg', format='svg')
print("📸 Validation Confusion Matrix saved as SVG.")
plt.show()

# ============================================================
# STEP 5 – FINAL TRAINING (STAYING TRUE TO THE ORIGINAL CONFIGURATION)
# ============================================================
print("\n" + "=" * 60)
print("   TRAINING FINAL MODEL WITH ORIGINAL FIXED HYPERPARAMETERS (K=21)")
print("=" * 60)

# Training the final model on the scaled combined dataset using your explicit fixed hyperparameters
final_knn_model = KNeighborsClassifier(n_neighbors=21, metric='manhattan', weights='distance', n_jobs=-1)
final_knn_model.fit(X_tuning_scaled, y_tuning)
print("✅ Final KNN model successfully trained on combined Train + Validation data.")

# ============================================================
# STEP 6 – EVALUATION ON THE OFFICIAL TEST DATASET
# ============================================================
print("Running predictions on the Official Test Dataset...")
y_test_pred = final_knn_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 50)
print(f"   🏆 OFFICIAL TEST DATASET ACCURACY: {test_accuracy:.2%}")
print("=" * 50)
print("\nFinal Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=le_final.classes_))

# ============================================================
# STEP 7 – SAVING FINAL TEST CONFUSION MATRIX
# ============================================================
print("\nGenerating performance plots for the test dataset...")
cm_test = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=le_final.classes_, yticklabels=le_final.classes_,
            ax=ax)
ax.set_title('Official Test Dataset - Confusion Matrix')
ax.set_ylabel('Actual Labels')
ax.set_xlabel('Predicted Labels')
plt.tight_layout()
plt.savefig('models/test_confusion_matrix_new.svg', format='svg', bbox_inches='tight')
plt.close()
print("✅ Test Confusion Matrix saved successfully as SVG in models folder!")

# ============================================================
# STEP 8 – EXPORTING PRODUCTION MODEL COMPONENTS
# ============================================================
print("\nSaving final model components for external deployment...")
joblib.dump(final_knn_model, 'models/best_posture_knn_model.pkl')
joblib.dump(imputer_final, 'models/posture_imputer.pkl')
joblib.dump(scaler_final, 'models/posture_scaler.pkl')
joblib.dump(le_final, 'models/posture_label_encoder.pkl')
print("✅ All production components saved successfully in the 'models/' folder!")

print("\n============ PIPELINE FINISHED SUCCESSFULLY ============")