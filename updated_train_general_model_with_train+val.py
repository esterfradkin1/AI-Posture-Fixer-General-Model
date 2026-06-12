import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

print("============ STARTING POSTURE PIPELINE (TRAIN+VAL -> TEST) ============")

# ============================================================
# STEP 1 – LOADING DATASETS (COMBINING TRAIN & VALIDATION)
# ============================================================
train_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\train_datasets"
val_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\validation_datasets"  # <--- נתיב תיקיית ה-Validation החדש
test_folder = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\test_datasets"


def load_folder_data(folder_path):
    """פונקציית עזר לטעינה דינמית של קבצים מתיקייה"""
    all_data = []
    if not os.path.exists(folder_path):
        print(f"⚠️ אזהרה: התיקייה {folder_path} לא קיימת.")
        return pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".csv"):
            file_full_path = os.path.join(folder_path, filename)

            if filename.endswith(".xlsx"):
                temp_df = pd.read_excel(file_full_path)
            else:
                temp_df = pd.read_csv(file_full_path)

            all_data.append(temp_df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


print("טוען נתוני אימון (Train)...")
train_df = load_folder_data(train_folder)

print("טוען נתוני תיקוף (Validation)...")
val_df = load_folder_data(val_folder)

print("טוען נתוני טסט (Test)...")
test_df = load_folder_data(test_folder)

if train_df.empty or test_df.empty:
    raise RuntimeError("חובה שגם תיקיית ה-train וגם תיקיית ה-test יכילו קובצי נתונים תקינים!")

# --- איחוד נתוני ה-Train וה-Validation יחד ---
print(f"מבצע איחוד: Train ({len(train_df)} שורות) + Validation ({len(val_df)} שורות)...")
combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"סך הכל דגימות בסט האימון המאוחד: {len(combined_train_df)}")

target_column = 'Label'
drop_cols = [target_column, 'Image_ID', 'Dataset_Split']

# התאמת ה-Label Encoder על בסיס נתוני האימון המאוחדים
le = LabelEncoder()
y_train_combined = le.fit_transform(combined_train_df[target_column])
y_test = le.transform(test_df[target_column])

# הפרדת הפיצ'רים מהתווית מתוך הסט המאוחד והטסט
X_train_combined = combined_train_df.drop(columns=drop_cols, errors='ignore')
X_test = test_df.drop(columns=drop_cols, errors='ignore')

# ============================================================
# STEP 2 – IMPUTATION & SCALING (TRAIN+VAL FIT)
# ============================================================
# המודל לומד את הסטטיסטיקות (ממוצע, חציון) מתוך סט האימון המאוחד
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_combined)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ============================================================
# STEP 3 – MODEL TRAINING (K=21)
# ============================================================
print(f"\nמאמן את מודל ה-KNN (K=21) על סט האימון המאוחד ({len(X_train_scaled)} דגימות)...")
best_knn_model = KNeighborsClassifier(
    n_neighbors=21,
    metric='manhattan',
    weights='distance',
    n_jobs=-1
)

# האימון מתבצע על הסט המאוחד (Train + Val)
best_knn_model.fit(X_train_scaled, y_train_combined)

# ============================================================
# STEP 4 – EVALUATION ON THE TEST DATASET
# ============================================================
print("מריץ תחזיות על ה-Test Dataset...")
y_test_pred = best_knn_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 50)
print(f"   🏆 OFFICIAL TEST DATASET ACCURACY: {test_accuracy:.2%}")
print("=" * 50)
print("\nFinal Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# ============================================================
# STEP 5 – SAVING CONFUSION MATRIX
# ============================================================
print("\nמפיק גרפי ביצועים לתיקיית models...")
os.makedirs('models', exist_ok=True)

cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title('Official Test Dataset - Confusion Matrix')
ax.set_ylabel('Actual Labels')
ax.set_xlabel('Predicted Labels')
plt.tight_layout()
plt.savefig('models/test_confusion_matrix_new.svg', format='svg', bbox_inches='tight')
plt.close()

print("✅ מטריצת הבלבול נשמרה בהצלחה בתיקיית models!")

# ============================================================
# STEP 6 – EXPORTING MODEL COMPONENTS
# ============================================================
print("\nשומר את רכיבי המודל הסופיים לשימוש עתידי...")
joblib.dump(best_knn_model, 'models/best_posture_knn_model.pkl')
joblib.dump(imputer, 'models/posture_imputer.pkl')
joblib.dump(scaler, 'models/posture_scaler.pkl')
joblib.dump(le, 'models/posture_label_encoder.pkl')
print("✅ כל הרכיבים נשמרו בהצלחה בתיקיית 'models/'!")

print("\n============ PIPELINE FINISHED SUCCESSFULLY ============")