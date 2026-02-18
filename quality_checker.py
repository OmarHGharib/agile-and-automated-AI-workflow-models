import numpy as np
import cv2

class QualityInspector:
    def __init__(self, blur_threshold=100.0, snr_threshold=5.0):
        # Thresholds need calibration:
        # Blur < 100 usually means "Motion Blur" or "Bad Resolution"
        # SNR < 5 usually means "Very Grainy"
        self.blur_threshold = blur_threshold
        self.snr_threshold = snr_threshold

    def inspect(self, patient_dict):
        """
        Input: The dictionary from 'MedicalResampler'
        Output: The same dict, but with a new 'qa_report' key.
        """
        if patient_dict["status"] != "Success":
            return patient_dict

        # We inspect the MIDDLE SLICE of the 3D volume
        # (The edges are often just empty air, which gives false results)
        data = patient_dict["data"]
        z_center = data.shape[2] // 2
        # Normalize to 0-255 for OpenCV (CV2 expects images, not raw float data)
        slice_img = self._normalize_for_cv2(data[:, :, z_center])

        # --- TEST 1: BLUR DETECTION ---
        # Logic: A sharp image has high variance in edges. Blurry = Low variance.
        laplacian_var = cv2.Laplacian(slice_img, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.blur_threshold

        # --- TEST 2: NOISE DETECTION (Blind SNR) ---
        # Logic: Compare the "Signal" (Center of brain) to "Noise" (Air in corner)
        snr_value = self._calculate_snr(slice_img)
        is_noisy = snr_value < self.snr_threshold

        # --- THE VERDICT ---
        # Fail if either test fails
        qa_status = "Pass"
        if is_blurry or is_noisy:
            qa_status = "Fail"

        # Attach the report to the patient data
        patient_dict["qa_report"] = {
            "status": qa_status,
            "blur_score": round(laplacian_var, 2),
            "is_blurry": is_blurry,
            "snr_score": round(snr_value, 2),
            "is_noisy": is_noisy
        }
        
        print(f"   -> QA Inspection: {qa_status} | Blur: {laplacian_var:.1f} | SNR: {snr_value:.1f}")
        return patient_dict

    def _calculate_snr(self, image):
        """
        Estimates Signal-to-Noise Ratio (SNR) without a reference image.
        """
        h, w = image.shape
        
        # 1. Estimate Noise from the top-left corner (assumed to be Air)
        # We take a 20x20 pixel box
        background_patch = image[0:20, 0:20]
        noise_sigma = np.std(background_patch)
        
        # Avoid division by zero if background is perfectly clean
        if noise_sigma == 0: 
            noise_sigma = 0.001

        # 2. Estimate Signal from the center (assumed to be Tissue)
        center_patch = image[h//2-20:h//2+20, w//2-20:w//2+20]
        signal_mu = np.mean(center_patch)

        # 3. SNR Formula
        return signal_mu / noise_sigma

    def _normalize_for_cv2(self, slice_data):
        """
        Helper: OpenCV requires 0-255 integers. Medical data is crazy floats.
        We Min-Max scale it to 0-255 just for the QA check.
        """
        mn = np.min(slice_data)
        mx = np.max(slice_data)
        if mx - mn == 0: return np.zeros_like(slice_data, dtype=np.uint8)
        
        norm = (slice_data - mn) / (mx - mn) * 255
        return norm.astype(np.uint8)

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    inspector = QualityInspector()
    
    # 1. Simulate a Good Scan
    good_scan = {
        'status': 'Success', 
        'data': np.random.normal(loc=100, scale=10, size=(200, 200, 50)) # Distinct signal
    }
    # Fix the corners to be "Air" (0 value) so SNR works
    good_scan['data'][0:20, 0:20, :] = 0 
    
    print("--- Inspecting Good Scan ---")
    result_good = inspector.inspect(good_scan)

    # 2. Simulate a Noisy Scan
    bad_scan = {
        'status': 'Success', 
        'data': np.random.normal(loc=50, scale=40, size=(200, 200, 50)) # High variance everywhere
    }
    print("\n--- Inspecting Noisy Scan ---")
    result_bad = inspector.inspect(bad_scan)