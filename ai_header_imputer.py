import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your model skeleton
from train_ultimate_ai import UltimateMedicalAI 

class AIHeaderImputer:
    def __init__(self, weights_path="ultimate_4task_ai.pth", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AI on: {self.device}")

        # --- YOUR CHOSEN ARCHITECTURE ---
        # 2 Modalities
        self.modality_map = {0: "MRI", 1: "CT"} 
        
        # 4 Body Parts
        self.body_part_map = {0: "Brain", 1: "Chest", 2: "Abdomen", 3: "Pelvis", 4: "Spine", 5: "Knee"}
        
        # 3 Planes
        self.plane_map = {0: "Axial", 1: "Coronal", 2: "Sagittal"}

        # Initialize the skeleton to perfectly match your maps
        self.model = UltimateMedicalAI(
            num_modalities=len(self.modality_map), 
            num_body_parts=len(self.body_part_map), 
            num_planes=len(self.plane_map)
        ) 
        
        # Load the weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_header(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert('RGB') 
        else:
            img_norm = ((image_path_or_array - image_path_or_array.min()) / 
                        (image_path_or_array.max() - image_path_or_array.min() + 1e-8) * 255)
            image = Image.fromarray(img_norm.astype(np.uint8)).convert('RGB')

        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_modality, pred_body_part, pred_plane, pred_spacing = self.model(input_tensor)

        modality_idx = torch.argmax(F.softmax(pred_modality, dim=1), dim=1).item()
        body_part_idx = torch.argmax(F.softmax(pred_body_part, dim=1), dim=1).item()
        plane_idx = torch.argmax(F.softmax(pred_plane, dim=1), dim=1).item()
        spacing_val = pred_spacing[0][0].item() 

        predicted_header = {
            "modality": self.modality_map.get(modality_idx, "Unknown"),
            "body_part": self.body_part_map.get(body_part_idx, "Unknown"),
            "image_plane": self.plane_map.get(plane_idx, "Unknown"),
            "pixel_spacing": [spacing_val, spacing_val, 1.0], 
            "is_rescued_by_ai": True
        }

        return predicted_header

if __name__ == "__main__":
    print("\n--- AI Header Imputer successfully loaded! ---")