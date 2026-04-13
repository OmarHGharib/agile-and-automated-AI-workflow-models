import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import your stations
from universal_loader import UniversalLoader
from smart_resampler import MedicalResampler
from normalizer import MedicalNormalizer

def try_pipeline(sample_path):
    print(f"--- STARTING PIPELINE TRIAL ---")
    print(f"Input: {os.path.basename(sample_path)}")

    # 1. Initialize Stations
    # 'ultimate_4task_ai.pth' is in folder
    loader = UniversalLoader(ai_weights_path="ultimate_4task_ai.pth")
    resampler = MedicalResampler(target_spacing=[1.0, 1.0, 1.0])
    normalizer = MedicalNormalizer()

    # --- STATION A: LOAD & AI RESCUE ---
    print("\n[Step 1] Loading & AI Metadata Imputation...")
    result = loader.load_data(sample_path)
    if result["status"] == "Error":
        print(f"Pipeline Failed at Loader: {result['message']}")
        return

    print(f"   -> AI Predicted Modality: {result.get('modality')}")
    print(f"   -> AI Predicted Spacing: {result.get('spacing')}")
    
    raw_data = result["data"]
    print(f"   -> Raw Shape: {raw_data.shape} | Max Value: {np.max(raw_data)}")

    # --- STATION B: RESAMPLING ---
    print("\n[Step 2] Resampling to 1.0mm cube...")
    result = resampler.process(result)
    resampled_data = result["data"]
    print(f"   -> New Shape: {resampled_data.shape}")

    # --- STATION C: NORMALIZATION ---
    print("\n[Step 3] Normalizing Pixels...")
    result = normalizer.process(result)
    final_data = result["data"]
    print(f"   -> Final Range: {np.min(final_data):.2f} to {np.max(final_data):.2f}")

    # --- VISUALIZATION ---
    print("\n[Step 4] Visualizing Results...")
    visualize_results(raw_data, final_data)

def visualize_results(original, processed):
    plt.figure(figsize=(12, 5))

    # Original (Middle Slice)
    plt.subplot(1, 2, 1)
    idx_orig = original.shape[-1] // 2 if original.ndim == 3 else 0
    slice_orig = original[:, :, idx_orig] if original.ndim == 3 else original
    plt.imshow(slice_orig, cmap='gray')
    plt.title("Original (Raw)")
    plt.colorbar()

    # Processed (Middle Slice)
    plt.subplot(1, 2, 2)
    idx_proc = processed.shape[-1] // 2 if processed.ndim == 3 else 0
    slice_proc = processed[:, :, idx_proc] if processed.ndim == 3 else processed
    plt.imshow(slice_proc, cmap='gray')
    plt.title("Pipeline Output (Ready for AI)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point this to your real JPEG or Zip file!
    test_file = "/Users/omargharib/Desktop/A stage/archive (1)/Data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000000 (6).png" 
    
    try_pipeline(test_file)