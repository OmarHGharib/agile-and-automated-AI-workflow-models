import numpy as np
import scipy.ndimage

class MedicalResampler:
    def __init__(self, target_spacing=[1.0, 1.0, 1.0]):
        # Target: 1mm x 1mm x 1mm voxels
        self.target_spacing = np.array(target_spacing)

    def process(self, patient_dict, force_mask=False):
        """
        Smart Resampling: Automatically chooses the right math for Images vs. Masks.
        """
        if patient_dict["status"] != "Success":
            return patient_dict

        data = patient_dict["data"]
        spacing = np.array(patient_dict["spacing"])
        
        # 1. Calculate Zoom Factor
        resize_factor = spacing / self.target_spacing

        # 2. DECISION LOGIC: Image vs. Mask
        # If user says it's a mask, OR if data looks like a mask (only 0s and 1s)
        unique_vals = np.unique(data)
        is_binary_mask = (len(unique_vals) <= 2) and (unique_vals.dtype == int or unique_vals.dtype == bool)
        
        if force_mask or is_binary_mask:
            # MASK MODE: Use Nearest Neighbor (Order 0)
            # Keeps edges sharp. 0 stays 0. 1 stays 1.
            print(f"   -> Mode: MASK (Nearest Neighbor) | Spacing: {spacing} -> {self.target_spacing}")
            resampled_data = scipy.ndimage.zoom(data, zoom=resize_factor, order=0)
        else:
            # IMAGE MODE: Use Cubic Spline (Order 3)
            # Keeps anatomy smooth.
            print(f"   -> Mode: IMAGE (Cubic Spline) | Spacing: {spacing} -> {self.target_spacing}")
            resampled_data = scipy.ndimage.zoom(data, zoom=resize_factor, order=3)

        # 3. Update the dictionary
        patient_dict["data"] = resampled_data
        patient_dict["spacing"] = self.target_spacing
        patient_dict["original_shape"] = data.shape
        patient_dict["new_shape"] = resampled_data.shape
        
        return patient_dict

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    resampler = MedicalResampler()

    # Scenario A: The MRI Scan (Continuous data)
    mri_scan = {
        'status': 'Success', 
        'data': np.random.rand(100, 100, 50), # Random float values
        'spacing': [0.5, 0.5, 2.0]
    }
    
    # Scenario B: The Tumor Mask (Binary data)
    # Creating a fake mask with only 0 and 1
    mask_data = np.zeros((100, 100, 50))
    mask_data[40:60, 40:60, :] = 1 
    tumor_mask = {
        'status': 'Success', 
        'data': mask_data, 
        'spacing': [0.5, 0.5, 2.0]
    }

    print("--- Processing MRI ---")
    res_mri = resampler.process(mri_scan)

    print("\n--- Processing Tumor Mask ---")
    res_mask = resampler.process(tumor_mask)
    
    # Verification
    print(f"\nMask Unique Values After Resampling: {np.unique(res_mask['data'])}")
    # If this prints [0. 1.], you succeeded. 
    # If it prints [0. 0.1 0.4 ...], you failed (and destroyed the label).