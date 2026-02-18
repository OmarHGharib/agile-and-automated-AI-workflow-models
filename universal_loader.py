import os
import pydicom
import nibabel as nib
import numpy as np
from PIL import Image
import patoolib  # <--- The Universal Archive Extractor
import tempfile
import shutil
import glob

class UniversalLoader:
    def __init__(self):
        self.default_spacing = [1.0, 1.0, 1.0]

    def load_data(self, input_path):
        """
        The "One-Button" Function: 
        Accepts Files (dcm, nii, jpg), Archives (rar, zip, 7z), or Folders.
        """
        if not os.path.exists(input_path):
            return {"status": "Error", "message": "Path not found"}

        # 1. If it's a FOLDER, scan it directly
        if os.path.isdir(input_path):
            return self._handle_folder(input_path)

        # 2. If it's a FILE, check extension
        ext = input_path.lower()
        
        # List of archive formats supported by patool
        archive_exts = (".zip", ".rar", ".7z", ".tar", ".gz", ".xz")
        
        if input_path.endswith(archive_exts):
            return self._handle_archive(input_path)
            
        elif ext.endswith(".dcm"):
            return self._handle_dicom_series([input_path]) # Treat as single-slice series
            
        elif ext.endswith((".nii", ".nii.gz")):
            return self._handle_nifti(input_path)
            
        elif ext.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            return self._handle_standard_image(input_path)
            
        else:
            return {"status": "Error", "message": f"Unsupported format: {ext}"}

    # --- THE GENERIC ARCHIVE HANDLER (RAR, ZIP, 7Z, TAR) ---
    def _handle_archive(self, archive_path):
        try:
            # Create a safe temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Extracting {os.path.basename(archive_path)}...")
                
                # patoolib automatically detects format (RAR, ZIP, etc.) and extracts
                try:
                    patoolib.extract_archive(archive_path, outdir=temp_dir, verbosity=-1)
                except Exception as e:
                    return {"status": "Error", "message": f"Extraction failed. Is WinRAR/7Zip installed? Error: {e}"}
                
                # Once extracted, treat it exactly like a normal Folder
                return self._handle_folder(temp_dir)

        except Exception as e:
            return {"status": "Error", "message": str(e)}

    # --- THE GENERIC FOLDER SCANNER ---
    def _handle_folder(self, folder_path):
        """
        Recursively scans a folder (or extracted archive) for the best available medical data.
        """
        # Priority 1: NIfTI (Gold Standard)
        nifti_files = glob.glob(os.path.join(folder_path, "**", "*.nii*"), recursive=True)
        if nifti_files:
            print(" -> Found NIfTI data.")
            return self._handle_nifti(nifti_files[0])
        
        # Priority 2: DICOM Series (Standard Clinical)
        dicom_files = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
        if dicom_files:
            print(f" -> Found {len(dicom_files)} DICOM slices.")
            return self._handle_dicom_series(dicom_files)

        # Priority 3: Standard Images (Fallback/JPEGs)
        img_files = glob.glob(os.path.join(folder_path, "**", "*.[jJ][pP][gG]"), recursive=True)
        if img_files:
            print(" -> Found standard images (JPEG/PNG).")
            # NOTE: If there are multiple JPEGs, we might want to load them as a volume
            # For now, let's load the first one or you can adapt to load all like DICOMs
            return self._handle_standard_image(img_files[0])

        return {"status": "Error", "message": "No valid medical images found in location."}

    # --- SPECIFIC HANDLERS (Same as before) ---
    def _handle_dicom_series(self, file_list):
        # (Same logic: Read files, Sort by Z-position, Stack into 3D Array)
        try:
            slices = [pydicom.dcmread(f) for f in file_list]
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) # Sort by depth
            image_3d = np.stack([s.pixel_array.astype(float) for s in slices], axis=-1)
            
            # Metadata extraction
            ref = slices[0]
            spacing = ref.get("PixelSpacing", None) # [x, y]
            thick = ref.get("SliceThickness", 1.0)
            
            if spacing:
                final_spacing = [float(spacing[0]), float(spacing[1]), float(thick)]
                is_rescued = False
            else:
                final_spacing = self._rescue_spacing(image_3d, "CT") # Default to CT logic
                is_rescued = True

            return {
                "status": "Success", 
                "source": "DICOM Volume", 
                "data": image_3d, 
                "spacing": final_spacing,
                "rescued": is_rescued
            }
        except Exception as e:
            return {"status": "Error", "message": str(e)}

    def _handle_nifti(self, path):
        # (Same logic: Nibabel load)
        img = nib.load(path)
        header = img.header
        zooms = header.get_zooms()[:3]
        return {"status": "Success", "source": "NIfTI", "data": img.get_fdata(), "spacing": zooms, "rescued": False}

    def _handle_standard_image(self, path):
        # (Same logic: PIL load + Rescue)
        img = np.array(Image.open(path).convert('L')).astype(float)
        return {
            "status": "Success", 
            "source": "Image", 
            "data": img, 
            "spacing": self._rescue_spacing(img, "MR"), 
            "rescued": True
        }

    def _rescue_spacing(self, array, modality="CT"):
        # (Same heuristic logic)
        dims = array.shape[:2]
        fov = 500.0 if modality == "CT" else 240.0
        return [fov/dims[1], fov/dims[1], 1.0]

# --- USAGE ---
if __name__ == "__main__":
    loader = UniversalLoader()
    
    # 1. Test on a RAR file
    print(loader.load_data(r"C:\Users\Omar\Desktop\stage m2\001\t0"))
    
    # 2. Test on a ZIP file
    print(loader.load_data(r"C:\Users\Omar\Desktop\data.zip"))
    
    # 3. Test on a Folder
    print(loader.load_data(r"C:\Users\Omar\Desktop\unzipped_data"))