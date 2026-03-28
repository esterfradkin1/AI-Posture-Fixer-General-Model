import pandas as pd
import os

# 1. Define files using join for safety
base_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
            r"\All individual Datasets - excel sheets\train_datasets")

files = [
    os.path.join(base_dir, "Output_Ester_Fradkin_labeling_first_video_Dataset.xlsx"),
    os.path.join(base_dir, "Output_Shirel_labelling_for_cvat_Dataset.xlsx"),
    os.path.join(base_dir, "Output_Shirel1_good_and_bad_posture_Dataset.xlsx"),
    os.path.join(base_dir, "Output_Shirel2_good_posture_Dataset.xlsx")
]

# 2. IMPORTANT: output_path MUST include the .xlsx filename!
output_path = os.path.join(base_dir, "Final_Train_Posture_Dataset_combined.xlsx")

# 3. Check which files actually exist
missing_files = [f for f in files if not os.path.exists(f)]
if missing_files:
    print("❌ The following files were not found:")
    for m in missing_files:
        print(f"   - {m}")
    print("\nCheck the folder to see if the names match exactly!")
else:
    # 4. Load and Combine
    print("Files found! Combining...")
    dfs = [pd.read_excel(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    # 5. Save
    try:
        combined.to_excel(output_path, index=False)
        print(f"✅ Combined dataset saved to:\n{output_path}")
    except PermissionError:
        print(f"❌ ERROR: Please close '{output_path}' in Excel first!")