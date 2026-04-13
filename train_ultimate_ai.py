import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. THE 4-TASK DATASET LOADER
# ==========================================
class UltimateMedicalDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['FileName'])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Extract our 4 Labels
        modality = torch.tensor(row['Modality_Encoded'], dtype=torch.long)
        body_part = torch.tensor(row['BodyPart_Encoded'], dtype=torch.long)
        image_plane = torch.tensor(row['Plane_Encoded'], dtype=torch.long)
        
        space_x = float(row['PixelSpacing_X']) if pd.notna(row['PixelSpacing_X']) else float('nan')
        space_y = float(row['PixelSpacing_Y']) if pd.notna(row['PixelSpacing_Y']) else float('nan')
        spacing = torch.tensor([space_x, space_y], dtype=torch.float32)

        return image, modality, body_part, image_plane, spacing

# ==========================================
# 2. THE 4-HEAD NEURAL NETWORK
# ==========================================
class UltimateMedicalAI(nn.Module):
    def __init__(self, num_modalities, num_body_parts, num_planes):
        super(UltimateMedicalAI, self).__init__()
        
        # The Vision Backbone
        self.backbone = models.resnet18(weights='DEFAULT')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove default head
        
        # The 4 Custom Brains (Heads)
        self.head_modality = nn.Linear(num_features, num_modalities)
        self.head_bodypart = nn.Linear(num_features, num_body_parts)
        self.head_plane = nn.Linear(num_features, num_planes)
        self.head_spacing = nn.Linear(num_features, 2)  

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.head_modality(features),
            self.head_bodypart(features),
            self.head_plane(features),
            self.head_spacing(features)
        )

# ==========================================
# 3. THE LEAK-PROOF TRAINING LOOP
# ==========================================
def train_model():
    print("Preparing Data and Encoders...")
    
    # 1. Load Data & Encode Text to Numbers automatically
    df = pd.read_csv("ultimate_dataset.csv")
    
    # Fill missing text with "UNKNOWN"
    df['Modality'] = df['Modality'].fillna('UNKNOWN')
    df['BodyPart'] = df['BodyPart'].fillna('UNKNOWN')
    df['ImagePlane'] = df['ImagePlane'].fillna('UNKNOWN')
    
    # Let sklearn map words to integers (e.g., AXIAL -> 0, CORONAL -> 1)
    le_mod = LabelEncoder()
    le_body = LabelEncoder()
    le_plane = LabelEncoder()
    
    df['Modality_Encoded'] = le_mod.fit_transform(df['Modality'])
    df['BodyPart_Encoded'] = le_body.fit_transform(df['BodyPart'])
    df['Plane_Encoded'] = le_plane.fit_transform(df['ImagePlane'])
    
    print(f"Detected {len(le_mod.classes_)} Modalities, {len(le_body.classes_)} Body Parts, and {len(le_plane.classes_)} Planes.")

    # 2. Leak-Proof Patient Split (80% Train, 20% Val)
    splitter = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    split = splitter.split(df, groups=df['PatientID'])
    train_inds, val_inds = next(split)
    
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    print(f"Leak-Proof Split: {len(train_df)} Train Images | {len(val_df)} Val Images\n")

    # 3. Setup Augmentations & DataLoaders
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(UltimateMedicalDataset(train_df, "ultimate_ai_ready_images", train_transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(UltimateMedicalDataset(val_df, "ultimate_ai_ready_images", val_transform), batch_size=16, shuffle=False)

    # 4. Initialize AI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltimateMedicalAI(len(le_mod.classes_), len(le_body.classes_), len(le_plane.classes_)).to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003) # Lower learning rate for complex tasks
    
    epochs = 20
    best_val_loss = float('inf')
    
    print(f"Starting 4-Task Training on {device}...\n")
    print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Modality %':<10} | {'BodyPart %':<10} | {'Plane %':<10}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        for images, mod, body, plane, space in train_loader:
            images, mod, body, plane, space = images.to(device), mod.to(device), body.to(device), plane.to(device), space.to(device)
            
            optimizer.zero_grad()
            pred_mod, pred_body, pred_plane, pred_space = model(images)
            
            loss_mod = criterion_class(pred_mod, mod)
            loss_body = criterion_class(pred_body, body)
            loss_plane = criterion_class(pred_plane, plane)
            
            valid_mask = ~torch.isnan(space[:, 0])
            loss_space = criterion_reg(pred_space[valid_mask], space[valid_mask]) if valid_mask.sum() > 0 else torch.tensor(0.0).to(device)
            
            total_loss = loss_mod + loss_body + loss_plane + (0.5 * loss_space)
            total_loss.backward()
            optimizer.step()
            
        # --- VALIDATION (MONITORING) ---
        model.eval()
        val_loss = 0
        corr_mod, corr_body, corr_plane, total_samples = 0, 0, 0, 0
        
        with torch.no_grad():
            for images, mod, body, plane, space in val_loader:
                images, mod, body, plane, space = images.to(device), mod.to(device), body.to(device), plane.to(device), space.to(device)
                
                pred_mod, pred_body, pred_plane, pred_space = model(images)
                
                # Math out the losses
                l_m = criterion_class(pred_mod, mod)
                l_b = criterion_class(pred_body, body)
                l_p = criterion_class(pred_plane, plane)
                
                valid_mask = ~torch.isnan(space[:, 0])
                l_s = criterion_reg(pred_space[valid_mask], space[valid_mask]) if valid_mask.sum() > 0 else torch.tensor(0.0).to(device)
                
                val_loss += (l_m + l_b + l_p + (0.5 * l_s)).item()
                
                # Tally the correct answers for accuracy tracking
                corr_mod += (torch.argmax(pred_mod, dim=1) == mod).sum().item()
                corr_body += (torch.argmax(pred_body, dim=1) == body).sum().item()
                corr_plane += (torch.argmax(pred_plane, dim=1) == plane).sum().item()
                total_samples += mod.size(0)
                
        # Calculate final epoch metrics
        avg_val_loss = val_loss / len(val_loader)
        acc_mod = (corr_mod / total_samples) * 100
        acc_body = (corr_body / total_samples) * 100
        acc_plane = (corr_plane / total_samples) * 100
        
        print(f"[{epoch+1:02d}/{epochs:02d}] | {avg_val_loss:<10.4f} | {acc_mod:<9.2f}% | {acc_body:<9.2f}% | {acc_plane:<9.2f}%")
        
        # Save Best Model dynamically
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "ultimate_4task_ai.pth")
            
    print("\nTraining Complete! Your Ultimate 4-Task AI is saved as 'ultimate_4task_ai.pth'.")

if __name__ == "__main__":
    train_model()