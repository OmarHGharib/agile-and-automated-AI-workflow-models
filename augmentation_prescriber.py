import numpy as np
import collections

class AugmentationPrescriber:
    def __init__(self, min_small_object_size=10, balance_threshold=0.2):
        # min_small_object_size: diameter in pixels below which an object is "Small"
        # balance_threshold: if minority class is < 20% of total, trigger balancing
        self.min_size = min_small_object_size
        self.threshold = balance_threshold

    def prescribe(self, batch_data_list):
        """
        Input: A list of dictionaries (one per patient) containing 'data' and 'mask'.
        Output: A report detailing which augmentations to turn ON.
        """
        # 1. Initialize Counters
        total_pixels = 0
        tumor_pixels = 0
        small_objects_found = 0
        total_objects = 0
        scanner_intensities = []

        # 2. Analyze the Batch
        for patient in batch_data_list:
            if "mask" not in patient:
                continue # Skip if no label

            mask = patient["mask"]
            image = patient["data"]

            # --- CHECK A: Class Imbalance ---
            # Count tumor pixels vs background
            t_count = np.count_nonzero(mask)
            tumor_pixels += t_count
            total_pixels += mask.size

            # --- CHECK B: Small Object Bias ---
            # Use connected components to find individual tumors
            # (We perform a quick check on the middle slice to save time)
            mid_slice = mask.shape[2] // 2
            # Simple object counting logic (you can use skimage.measure.label for 3D)
            if t_count > 0:
                # Estimate diameter from volume (assuming sphere V = 4/3 pi r^3)
                # This is a heuristic. 
                # Better: Check bounding box diagonal
                pass 

            # --- CHECK C: Scanner Bias ---
            # Record the mean intensity of the brain (ignoring 0 background)
            mean_intensity = np.mean(image[image > 0])
            scanner_intensities.append(mean_intensity)

        # 3. Formulate the Prescription
        prescription = {
            "Oversampling": False,
            "CopyPaste": False,
            "IntensityShift": False,
            "Reason": []
        }

        # --- DIAGNOSIS A: Imbalance ---
        tumor_ratio = tumor_pixels / (total_pixels + 1e-9)
        if tumor_ratio < self.threshold:
            prescription["Oversampling"] = True
            prescription["Reason"].append(f"Severe Imbalance (Tumor Ratio: {tumor_ratio:.4f})")

        # --- DIAGNOSIS C: Scanner Variance ---
        # If standard deviation of intensities is high (> 50 HU or equivalent), scanners are different.
        if len(scanner_intensities) > 1:
            intensity_variance = np.std(scanner_intensities)
            if intensity_variance > 20.0: # Threshold depends on normalization
                prescription["IntensityShift"] = True
                prescription["Reason"].append(f"High Scanner Variance (Std: {intensity_variance:.1f})")

        return prescription

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    prescriber = AugmentationPrescriber()
    
    # Simulate a batch of 3 patients
    # Patient 1: Large tumor, Bright scanner
    p1 = {
        "data": np.random.normal(100, 10, (100,100,50)),
        "mask": np.zeros((100,100,50))
    }
    p1["mask"][40:60, 40:60, :] = 1 # Big tumor
    
    # Patient 2: No tumor, Dark scanner
    p2 = {
        "data": np.random.normal(50, 10, (100,100,50)),
        "mask": np.zeros((100,100,50))
    }
    
    # Patient 3: Tiny tumor (Small Object), Medium scanner
    p3 = {
        "data": np.random.normal(80, 10, (100,100,50)),
        "mask": np.zeros((100,100,50))
    }
    p3["mask"][50:52, 50:52, 25:27] = 1 # Tiny 2x2x2 tumor

    batch = [p1, p2, p3]
    
    rx = prescriber.prescribe(batch)
    print("--- AUGMENTATION PRESCRIPTION ---")
    print(f"Oversampling Needed? {rx['Oversampling']}")
    print(f"Intensity Aug Needed? {rx['IntensityShift']}")
    print(f"Reasons: {rx['Reason']}")