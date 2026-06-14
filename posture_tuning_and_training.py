import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PosturePipelineBlackBox:
    """
    A unified Black Box pipeline to load posture datasets, preprocess features,
    run hyperparameter grid search, analyze feature importance, evaluate,
    and export production-ready model components.
    """

    def __init__(self, train_folder, val_folder, test_folder, output_dir='models'):
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.output_dir = output_dir

        # Target details
        self.target_column = 'Label'
        self.drop_cols = [self.target_column, 'Image_ID', 'Dataset_Split']

        # Pipeline Components
        self.le = LabelEncoder()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.final_model = None
        self.feature_names = []

        # Data Placeholders
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.combined_tuning_df = pd.DataFrame()

        # Processed Arrays
        self.X_tuning_scaled = None
        self.y_tuning = None
        self.X_test_scaled = None
        self.y_test = None
        self.is_train_tuning = None

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_folder_data(self, folder_path, dataset_type=None):
        """Helper to dynamically load Excel or CSV files from a directory."""
        all_data = []
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: The folder {folder_path} does not exist.")
            return pd.DataFrame()

        for filename in os.listdir(folder_path):
            if filename.endswith(".xlsx") or filename.endswith(".csv"):
                file_full_path = os.path.join(folder_path, filename)
                temp_df = pd.read_excel(file_full_path) if filename.endswith(".xlsx") else pd.read_csv(file_full_path)

                if dataset_type is not None:
                    temp_df['Dataset_Split'] = dataset_type
                all_data.append(temp_df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def load_and_prepare_data(self):
        """Step 1 & 2: Load data and perform unified preprocessing/scaling."""
        print("\n📥 Loading datasets from folders...")
        self.train_df = self._load_folder_data(self.train_folder, "train")
        self.val_df = self._load_folder_data(self.val_folder, "val")
        self.test_df = self._load_folder_data(self.test_folder)

        if self.train_df.empty or self.val_df.empty or self.test_df.empty:
            raise RuntimeError("Train, Validation, and Test folders must all contain valid data files!")

        print(f"Merging: Train ({len(self.train_df)} rows) + Validation ({len(self.val_df)} rows)...")
        self.combined_tuning_df = pd.concat([self.train_df, self.val_df], ignore_index=True)

        # Encode Targets
        self.y_tuning = self.le.fit_transform(self.combined_tuning_df[self.target_column])
        self.y_test = self.le.transform(self.test_df[self.target_column])

        # Separate Features
        X_tuning = self.combined_tuning_df.drop(columns=self.drop_cols, errors='ignore')
        X_test = self.test_df.drop(columns=self.drop_cols, errors='ignore')
        self.feature_names = list(X_tuning.columns)

        # Fit Transform Imputer and Scaler on fully combined tuning data
        X_tuning_imputed = self.imputer.fit_transform(X_tuning)
        X_test_imputed = self.imputer.transform(X_test)

        self.X_tuning_scaled = self.scaler.fit_transform(X_tuning_imputed)
        self.X_test_scaled = self.scaler.transform(X_test_imputed)

        # Track training mask within the tuning set
        self.is_train_tuning = (self.combined_tuning_df['Dataset_Split'] == 'train').values
        print("✅ Data loading and preprocessing transformations completed.")

    def run_grid_search(self):
        """Step 3: Execute grid search using a PredefinedSplit."""
        print("\n🔍 Running Hyperparameter Tuning Grid Search...")
        split_indices = np.where(self.combined_tuning_df['Dataset_Split'] == 'train', -1, 0)
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
        grid_search.fit(self.X_tuning_scaled, self.y_tuning)

        print("!" * 50)
        print(f"GRID SEARCH BEST ACCURACY: {grid_search.best_score_:.2%}")
        print(f"GRID SEARCH BEST CONFIGURATION: {grid_search.best_params_}")
        print("!" * 50)
        return grid_search.cv_results_

    def generate_visualizations_and_val_report(self, cv_results=None, fixed_k=21, fixed_metric='manhattan',
                                               fixed_weights='distance'):
        """Step 4: Generate insights, plots, and analyze the validation split performance."""
        print("\n📊 Generating Visualizations and Validation Analysis...")
        X_train_only = self.X_tuning_scaled[self.is_train_tuning]
        y_train_only = self.y_tuning[self.is_train_tuning]
        X_val_only = self.X_tuning_scaled[~self.is_train_tuning]
        y_val_only = self.y_tuning[~self.is_train_tuning]

        # 1. Feature Importance Plot
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_only, y_train_only)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 12))
        sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
        plt.title('Complete Feature Importance Analysis (Train Dataset)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.svg', format='svg')
        plt.close()

        # 2. Grid Search Plot
        if cv_results is not None:
            results_df = pd.DataFrame(cv_results)
            results_df['param_n_neighbors'] = results_df['param_n_neighbors'].astype(int)
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=results_df, x='param_n_neighbors', y='mean_test_score', hue='param_metric',
                         style='param_weights', markers=True)
            plt.title("Grid Search Validation Accuracy vs. K")
            plt.ylabel("Accuracy on Validation Dataset")
            plt.xlabel("Number of Neighbors (K)")
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/grid_search_accuracy_vs_k.svg', format='svg')
            plt.close()

        # 3. Validation Report & Confusion Matrix
        val_model = KNeighborsClassifier(n_neighbors=fixed_k, metric=fixed_metric, weights=fixed_weights, n_jobs=-1)
        val_model.fit(X_train_only, y_train_only)
        y_val_pred = val_model.predict(X_val_only)

        print("\n" + "=" * 50)
        print("   VALIDATION DATASET CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(y_val_only, y_val_pred, target_names=self.le.classes_))

        cm_val = confusion_matrix(y_val_only, y_val_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=self.le.classes_,
                    yticklabels=self.le.classes_)
        plt.title('Confusion Matrix: Validation Dataset Results')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/validation_confusion_matrix.svg', format='svg')
        plt.close()
        print(f"📸 Visualizations safely stored as SVGs inside '{self.output_dir}/'.")

    def train_final_model(self, k=21, metric='manhattan', weights='distance'):
        """Step 5: Train final model on Train + Val combined data."""
        print(f"\n🏋️ Training final model with fixed hyperparameters (K={k}, Metric={metric})...")
        self.final_model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
        self.final_model.fit(self.X_tuning_scaled, self.y_tuning)
        print("✅ Final KNN model successfully trained on complete combined dataset.")

    def evaluate_on_test_set(self):
        """Step 6 & 7: Evaluate performance on the isolated test set and save matrix."""
        if self.final_model is None:
            raise RuntimeError("Final model must be trained before testing!")

        print("\n🎯 Running predictions on the Official Test Dataset...")
        y_test_pred = self.final_model.predict(self.X_test_scaled)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print("\n" + "=" * 50)
        print(f"   🏆 OFFICIAL TEST DATASET ACCURACY: {test_accuracy:.2%}")
        print("=" * 50)
        print(classification_report(self.y_test, y_test_pred, target_names=self.le.classes_))

        # Test Matrix Visual Saved directly
        cm_test = confusion_matrix(self.y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=self.le.classes_,
                    yticklabels=self.le.classes_, ax=ax)
        ax.set_title('Official Test Dataset - Confusion Matrix')
        ax.set_ylabel('Actual Labels')
        ax.set_xlabel('Predicted Labels')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/test_confusion_matrix_new.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("✅ Test Confusion Matrix saved successfully.")

    def export_production_components(self, filename='posture_knn_blackbox_new.joblib'):
        """
        Exports all pipeline components packed tightly into a single joblib file
        to be consumed directly by the live inference script.
        """
        if self.final_model is None:
            raise RuntimeError("Cannot export components before training the final model!")

        # Packing all pipeline elements into a unified dictionary structure
        pack = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "model": self.final_model,
            "feature_names": self.feature_names,
            "classes": list(self.le.classes_),
            "label_encoder": self.le
        }

        output_path = os.path.join(self.output_dir, filename)
        joblib.dump(pack, output_path)
        print(f"📦 Unified Black Box successfully packed and saved to: {output_path}")


# ============================================================
# HOW TO RUN THE BLACK BOX
# ============================================================
if __name__ == "__main__":
    print("============ STARTING POSTURE PIPELINE (BLACK BOX EXECUTION) ============")

    # Path initializations
    TRAIN_DIR = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\train_datasets"
    VAL_DIR = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\validation_datasets"
    TEST_DIR = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\All individual Datasets - excel sheets\test_datasets"

    # Instantiate the box
    pipeline = PosturePipelineBlackBox(train_folder=TRAIN_DIR, val_folder=VAL_DIR, test_folder=TEST_DIR)

    # Trigger execution sequence smoothly
    pipeline.load_and_prepare_data()
    cv_res = pipeline.run_grid_search()
    pipeline.generate_visualizations_and_val_report(cv_results=cv_res)
    pipeline.train_final_model(k=21, metric='manhattan', weights='distance')
    pipeline.evaluate_on_test_set()
    pipeline.export_production_components()

    print("\n============ PIPELINE FINISHED SUCCESSFULLY ============")