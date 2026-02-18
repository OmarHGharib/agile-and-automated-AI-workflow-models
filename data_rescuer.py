import os
import shutil
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

class DataRescuer:
    def __init__(self, quarantine_folder="./quarantine"):
        self.quarantine_folder = quarantine_folder
        os.makedirs(self.quarantine_folder, exist_ok=True)

    def guess_modality_by_histogram(self, image_array):
        """
        Strategy 1: Distinguish CT from MRI/X-Ray using pixel intensity ranges.
        Logic: CT scales are fixed (Air = -1000). MRI scales are arbitrary (start at 0).
        """
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        mean_val = np.mean(image_array)

        # CT Logic: Hounsfield Units often drop to -1000 (Air) or -2000 (Sensor edge)
        # MRI Logic: Usually unsigned integers (0 to 65535), rarely negative.
        if min_val < -500:
            return "CT"
        
        # Refinement: Sometimes CTs are shifted to positive values (offset +1024)
        # If the dynamic range is distinctively 'CT-like' (Air to Bone)
        if min_val >= 0 and (max_val - min_val) > 2000 and mean_val < 500:
            return "CT_shifted" # Requires subtraction of 1024 later

        if min_val >= 0:
            return "MR" # Default guess for non-negative medical images

        return "Unknown"

    def estimate_pixel_spacing_heuristic(self, image_array, modality="CT"):
        """
        Strategy 2: Guess pixel size based on standard Field of View (FOV).
        Logic: A standard body CT covers ~500mm across.
        """
        rows, cols = image_array.shape
        
        if modality == "CT" or modality == "CT_shifted":
            # Standard Body CT FOV is approx 500mm.
            # 512 pixels / 500 mm = ~0.97 mm per pixel.
            estimated_spacing = 500.0 / cols
            return [estimated_spacing, estimated_spacing]
        
        elif modality == "MR":
            # Brain MRI FOV is approx 240mm.
            estimated_spacing = 240.0 / cols
            return [estimated_spacing, estimated_spacing]

        # Default fallback
        return [1.0, 1.0]

    def detect_contrast_enhancement(self, image_array, modality="CT"):
        """
        Strategy 3: Check for bright blood vessels to detect contrast.
        Logic: In soft tissue (0-400 HU), contrast pushes vessels to >150 HU.
        Unenhanced blood is ~40 HU.
        """
        if "CT" not in modality:
            return "Unknown (MR logic pending)"

        # 1. Isolate the "Soft Tissue" range (ignore bone > 700 and air < -500)
        # This prevents bones from triggering a "False Positive" for contrast.
        soft_tissue_pixels = image_array[(image_array > 0) & (image_array < 700)]

        if len(soft_tissue_pixels) == 0:
            return False

        # 2. Check the 99th Percentile of the soft tissue
        p99 = np.percentile(soft_tissue_pixels, 99)

        # If the brightest soft tissue is > 130 HU, it is likely Contrast Enhanced.
        # (Normal muscle/organ is ~40-60 HU).
        if p99 > 130:
            return True
        return False

    def process_file(self, file_path):
        """
        The Main Orchestrator: Tries to read tags, falls back to rescue logic if failed.
        """
        try:
            # Attempt to read header
            ds = pydicom.dcmread(file_path)
            
            # --- CHECK 1: Modality ---
            modality = ds.get("Modality", None)
            imputed_modality = False
            
            if modality is None:
                # RUN RESCUE STRATEGY 1
                modality = self.guess_modality_by_histogram(ds.pixel_array)
                imputed_modality = True

            # --- CHECK 2: Pixel Spacing ---
            spacing = ds.get("PixelSpacing", None)
            imputed_spacing = False
            
            if spacing is None:
                # RUN RESCUE STRATEGY 2
                spacing = self.estimate_pixel_spacing_heuristic(ds.pixel_array, modality)
                imputed_spacing = True

            # --- CHECK 3: Contrast (Series Description) ---
            desc = ds.get("SeriesDescription", "").lower()
            contrast_guess = "No"
            
            # If tag is missing or ambiguous, check pixels
            if desc == "" or "unknown" in desc:
                # RUN RESCUE STRATEGY 3
                has_contrast = self.detect_contrast_enhancement(ds.pixel_array, modality)
                contrast_guess = "Yes" if has_contrast else "No"
            else:
                 # Standard text check
                if any(x in desc for x in ['+c', 'contrast', 'venous', 'arterial']):
                    contrast_guess = "Yes"

            return {
                "status": "Success",
                "modality": modality,
                "pixel_spacing": spacing,
                "has_contrast": contrast_guess,
                "imputed_tags": imputed_modality or imputed_spacing
            }

        except Exception as e:
            # If file is totally broken, move to Quarantine
            filename = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(self.quarantine_folder, filename))
            return {"status": "Quarantined", "error": str(e)}

# --- Usage Example ---
if __name__ == "__main__":
    rescuer = DataRescuer()
    
    # Simulate a file (replace with your actual path)
    result = rescuer.process_file("./your_data/sample_image.dcm")
    print(result)