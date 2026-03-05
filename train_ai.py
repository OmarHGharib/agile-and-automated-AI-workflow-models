import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. THE DATASET LOADER
# ==========================================
class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Map text modalities to numbers for the AI
        self.modality_map = {"CT": 0, "MR": 1, "DX": 2, "CR": 2} 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load Image
        img_path = os.path.join(self.img_dir, row['FileName'])
        image = Image.open(img_path).convert("RGB") # ResNet expects 3 color channels
        
        if self.transform:
            image = self.transform(image)
            
        # Get Modality Label (Default to 0 if unknown to prevent crashes)
        modality_str = str(row['Modality']).upper()
        modality_label = self.modality_map.get(modality_str, 0)
        
        # Get Pixel Spacing (Use NaN if missing)
        space_x = float(row['PixelSpacing_X']) if pd.notna(row['PixelSpacing_X']) else float('nan')
        space_y = float(row['PixelSpacing_Y']) if pd.notna(row['PixelSpacing_Y']) else float('nan')
        spacing_label = torch.tensor([space_x, space_y], dtype=torch.float32)

        return image, torch.tensor(modality_label, dtype=torch.long), spacing_label

# ==========================================
# 2. THE MULTI-TASK NEURAL NETWORK
# ==========================================
class MultiTaskMedicalAI(nn.Module):
    def __init__(self):
        super(MultiTaskMedicalAI, self).__init__()
        
        # Load a pre-trained ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Extract the number of features going into the final layer (512 for ResNet18)
        num_features = self.backbone.fc.in_features
        
        # Remove the original final layer
        self.backbone.fc = nn.Identity()
        
        # Build our two custom "Heads"
        self.modality_head = nn.Linear(num_features, 3) # 3 outputs: CT, MR, DX
        self.spacing_head = nn.Linear(num_features, 2)  # 2 outputs: Spacing X and Y

    def forward(self, x):
        features = self.backbone(x)
        modality_pred = self.modality_head(features)
        spacing_pred = self.spacing_head(features)
        return modality_pred, spacing_pred

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_model():
    print("Setting up data and model...")
    
    # Standardize images to 224x224 (ResNet's favorite size)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = MedicalDataset(csv_file="dataset.csv", img_dir="ai_ready_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # Process 8 images at a time
    
    # Initialize Model, Loss Functions, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskMedicalAI().to(device)
    
    criterion_modality = nn.CrossEntropyLoss()
    criterion_spacing = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print(f"Starting training on {device} for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, modalities, spacings in dataloader:
            images, modalities = images.to(device), modalities.to(device)
            spacings = spacings.to(device)
            
            # 1. Forward Pass (Make Predictions)
            optimizer.zero_grad()
            pred_modalities, pred_spacings = model(images)
            
            # 2. Calculate Modality Loss (Always happens)
            loss_modality = criterion_modality(pred_modalities, modalities)
            
            # 3. Calculate Spacing Loss (Only for images that HAVE spacing data)
            # We create a "mask" to find valid numbers and ignore NaNs
            valid_mask = ~torch.isnan(spacings[:, 0]) 
            
            if valid_mask.sum() > 0:
                loss_spacing = criterion_spacing(pred_spacings[valid_mask], spacings[valid_mask])
            else:
                loss_spacing = torch.tensor(0.0).to(device)
                
            # 4. Combine Losses and Backpropagate (Learn)
            loss = loss_modality + (0.5 * loss_spacing) # 0.5 is our lambda weight
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {total_loss/len(dataloader):.4f}")

    print("\nTraining Complete! Saving AI Brain...")
    torch.save(model.state_dict(), "medical_ai_model.pth")

if __name__ == "__main__":
    train_model()