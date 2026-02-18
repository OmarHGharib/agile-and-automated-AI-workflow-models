import os
import numpy as np
import pandas as pd
import time

# Import your custom blocks
from universal_loader import UniversalLoader
from smart_resampler import MedicalResampler
from quality_checker import QualityInspector
from augmentation_prescriber import AugmentationPrescriber

def run_pipeline(input_folder, output_csv="processing_report.csv"):
    print(f"--- STARTING AI DATA PIPELINE ---")
    print(f"Target Folder: {input_folder}")
    
    # 1. Initialize the Workers
    loader = UniversalLoader()
    resampler = MedicalResampler(target_spacing=[1.0, 1.0, 1.0]) # Force 1mm standard
    inspector = QualityInspector(blur_threshold=100.0, snr_threshold=5.0)
    prescriber = AugmentationPrescriber()

    # 2. Scan for files (Recursive)
    all_files = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            # We accept Zips and standard image formats
            if f.lower().endswith((".zip", ".dcm", ".nii", ".nii.gz", ".jpg", ".png")):
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} potential files/archives.\n")

    # 3. Process Loop
    processed_batch = []
    report_data = []

    for idx, file_path in enumerate(all_files):
        print(f"[{idx+1}/{len(all_files)}] Processing: {os.path.basename(file_path)}...")
        start_time = time.time()
        
        # --- STATION A: INGESTION ---
        # (Auto-detects format, unzips if needed, handles errors)
        data_dict = loader.load_data(file_path)
        
        if data_dict["status"] != "Success":
            print(f"   -> FAIL (Loader): {data_dict['message']}")
            report_data.append({"File": file_path, "Status": "Load_Fail", "Error": data_dict['message']})
            continue

        # --- STATION B: STANDARDIZATION ---
        # (Resamples to 1x1x1mm)
        # Note: We assume it's an Image for now. If you have masks, change force_mask=True
        data_dict = resampler.process(data_dict, force_mask=False)
        
        # --- STATION C: QUALITY CONTROL ---
        # (Checks for blur and noise)
        data_dict = inspector.inspect(data_dict)
        qa = data_dict["qa_report"]
        
        # Store results for Prescriber
        # (In a real run, you'd save the numpy array to disk here)
        processed_batch.append(data_dict)

        # Log the result
        status_msg = "OK" if qa["status"] == "Pass" else "QUARANTINE"
        print(f"   -> Result: {status_msg} (Blur: {qa['blur_score']}, SNR: {qa['snr_score']})")
        
        report_data.append({
            "File": os.path.basename(file_path),
            "Status": status_msg,
            "Original_Spacing": data_dict["spacing"], # From loader
            "Rescued_Metadata": data_dict.get("metadata", {}).get("is_rescued", "N/A"),
            "Blur_Score": qa["blur_score"],
            "SNR_Score": qa["snr_score"],
            "Processing_Time": round(time.time() - start_time, 2)
        })

    # --- STATION D: STRATEGY PRESCRIPTION ---
    print("\n--- GENERATING TRAINING STRATEGY ---")
    if processed_batch:
        # The prescriber analyzes the whole batch to find bias
        strategy = prescriber.prescribe(processed_batch)
        print(f"Prescribed Strategy: {strategy}")
    else:
        print("No valid data found to generate strategy.")

    # 4. Save the Report
    df = pd.DataFrame(report_data)
    df.to_csv(output_csv, index=False)
    print(f"\nPipeline Complete. Report saved to: {output_csv}")

# --- EXECUTION ---
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR REAL DATASET FOLDER
    dataset_path = r"C:\Users\Omar\Desktop\stage m2\001\t0" 
    
    run_pipeline(dataset_path)