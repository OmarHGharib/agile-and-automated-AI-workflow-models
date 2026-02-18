import numpy as np
import scipy.ndimage

class MedicalResampler:
    def __init__(self, target_spacing=[1.0, 1.0, 1.0]):
        # We target 1mm x 1mm x 1mm voxels (Standard for AI)
        self.target_spacing = np.array(target_spacing)

    def process_patient(self, patient_data_dict):
        """
        Input: The dictionary from UniversalLoader
        Output: A new dictionary with resized data
        """
        # 1. Check if data is valid
        if patient_data_dict["status"] != "Success":
            return patient_data_dict # Pass errors through

        image = patient_data_dict["data"] # The 3D array
        current_spacing = np.array(patient_data_dict["spacing"]) # e.g. [0.5, 0.5, 1.0]

        # 2. Calculate the "Zoom Factor"
        # Formula: Current_Size / Target_Size
        # Example: 0.5mm / 1.0mm = 0.5 (Shrink the array by half)
        resize_factor = current_spacing / self.target_spacing

        # 3. Perform the Resampling (Interpolation)
        # Order=3 (Cubic) is best for Images (smooth)
        # Order=0 (Nearest) is best for Masks (sharp edges)
        print(f"   -> Resampling from {current_spacing} to {self.target_spacing}...")
        
        new_image = scipy.ndimage.zoom(image, zoom=resize_factor, order=3)

        # 4. Update the Dictionary
        patient_data_dict["data"] = new_image
        patient_data_dict["spacing"] = self.target_spacing # Now it is standard!
        patient_data_dict["original_shape"] = image.shape
        patient_data_dict["new_shape"] = new_image.shape
        
        return patient_data_dict

# --- USAGE WITH YOUR LOADER ---
if __name__ == "__main__":
    # 1. Load the data (Station A)
    # Assuming 'loader' is your UniversalLoader instance from previous code
    # raw_result = loader.load_data(r"C:\Users\Omar\Desktop\stage m2\001\t0") 
    
    # Let's simulate the result you just got from the NIfTI file:
    simulated_result = {
        'status': 'Success',
        'data': np.random.rand(240, 240, 155), # Fake brain volume
        'spacing': [0.5, 0.5, 1.0]             # Your actual spacing
    }

    # 2. Resample the data (Station B)
    resampler = MedicalResampler(target_spacing=[1.0, 1.0, 1.0])
    clean_result = resampler.process_patient(simulated_result)

    print(f"\nOriginal Shape: {clean_result['original_shape']}") # Should be (240, 240, 155)
    print(f"New Shape:      {clean_result['new_shape']}")      # Should be (120, 120, 155)
    print("Status: Ready for AI Model")