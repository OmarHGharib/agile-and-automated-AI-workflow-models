import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image

master_folder = "my_merged_ai_dataset/"
output_folder = "ai_ready_images/"
dataset_labels = []

# Create a new folder for the standard images
os.makedirs(output_folder, exist_ok=True)

print("Scanning DICOMs, extracting metadata, and converting to PNG...")

for root, dirs, files in os.walk(master_folder):
    for file in files:
        if file.endswith(".dcm"):
            dcm_path = os.path.join(root, file)
            
            try:
                # 1. Read the DICOM file (with pixels included this time!)
                dicom_data = pydicom.dcmread(dcm_path)
                
                # 2. Extract the Metadata (The Ground Truth for the AI)
                modality = dicom_data.get('Modality', 'Unknown')
                pixel_spacing = dicom_data.get('PixelSpacing', [None, None])
                
                # 3. Process the Image Pixels
                # Convert raw medical pixels into a standard mathematical array
                image_array = dicom_data.pixel_array.astype(float)
                
                # Normalize the pixels to standard 0-255 format so it looks right as a PNG
                if image_array.max() > 0:
                    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
                
                image_array = np.uint8(image_array)
                
                # Save as a PNG
                png_filename = file.replace(".dcm", ".png")
                png_path = os.path.join(output_folder, png_filename)
                
                img = Image.fromarray(image_array)
                img.save(png_path)
                
                # 4. Save the labels for the AI's "Answer Key"
                dataset_labels.append({
                    "FileName": png_filename,
                    "Modality": modality,
                    "PixelSpacing_X": pixel_spacing[0],
                    "PixelSpacing_Y": pixel_spacing[1] if len(pixel_spacing) > 1 else None
                })
                
            except Exception as e:
                # If a file is a weird "scout" scan without proper pixels, skip it gracefully
                pass

# 5. Save the master CSV
df = pd.DataFrame(dataset_labels)
csv_path = "dataset.csv"
df.to_csv(csv_path, index=False)

print(f"\nSuccess! Converted {len(df)} images.")
print(f"Check the '{output_folder}' folder for your images and '{csv_path}' for your labels.")
print("\nHere is a peek at your AI's new training data:")
print(df.head())
