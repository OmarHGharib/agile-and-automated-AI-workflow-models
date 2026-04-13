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
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. THE DATASET LOADER
# ==========================================
class UniverseDataset(Dataset):
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
            
        modality = torch.tensor(row['Modality_Encoded'], dtype=torch.long)
        body_part = torch.tensor(row['BodyPart_Encoded'], dtype=torch.long)
        image_plane = torch.tensor(row['Plane_Encoded'], dtype=torch.long)
        
        space_x = float(row['PixelSpacing_X']) if pd.notna(row['PixelSpacing_X']) else float('nan')
        space_y = float(row['PixelSpacing_Y']) if pd.notna(row['PixelSpacing_Y']) else float('nan')
        spacing = torch.tensor([space_x, space_y], dtype=torch.float32)

        return image, modality, body_part, image_plane, spacing

# ==========================================
# 2. THE MULTI-TASK NETWORK
# ==========================================
class UniverseMedicalAI(nn.Module):
    def __init__(self, num_modalities, num_body_parts, num_planes):
        super(UniverseMedicalAI, self).__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
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
# 3. THE TRAINING LOOP & FINAL EXAM
# ==========================================
def train_model():
    print("🚀 Loading Universe Data (37,000+ Images)...")
    
    # 1. Load Data
    df = pd.read_csv("universe_dataset.csv")
    
    # Ensure no missing text crashes the encoders
    df['Modality'] = df['Modality'].fillna('UNKNOWN').astype(str)
    df['BodyPart'] = df['BodyPart'].fillna('UNKNOWN').astype(str).str.upper()
    df['ImagePlane'] = df['ImagePlane'].fillna('UNKNOWN').astype(str)
    
    # 🚨 SEMANTIC MERGING: Fix the rare classes by merging them into parent anatomies 🚨
    df['BodyPart'] = df['BodyPart'].replace({
        'LUNG': 'CHEST',
        'TSPINE': 'CHEST',
        'LEG': 'EXTREMITY'
    })
    
    le_mod = LabelEncoder().fit(df['Modality'])
    le_body = LabelEncoder().fit(df['BodyPart'])
    le_plane = LabelEncoder().fit(df['ImagePlane'])
    
    df['Modality_Encoded'] = le_mod.transform(df['Modality'])
    df['BodyPart_Encoded'] = le_body.transform(df['BodyPart'])
    df['Plane_Encoded'] = le_plane.transform(df['ImagePlane'])

    print(f"🧬 Consolidated to {len(le_body.classes_)} distinct Body Parts: {list(le_body.classes_)}")

    # 2. THE 3-WAY LEAK-PROOF SPLIT (70% Train, 15% Val, 15% Test)
    splitter_1 = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=42)
    train_inds, temp_inds = next(splitter_1.split(df, groups=df['PatientID']))
    
    train_df = df.iloc[train_inds]
    temp_df = df.iloc[temp_inds]
    
    splitter_2 = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=42)
    val_inds, test_inds = next(splitter_2.split(temp_df, groups=temp_df['PatientID']))
    
    val_df = temp_df.iloc[val_inds]
    test_df = temp_df.iloc[test_inds]
    
    print(f"\n📊 DATASET SPLIT:")
    print(f"  - Training Set   : {len(train_df)} Images")
    print(f"  - Validation Set : {len(val_df)} Images")
    print(f"  - Secret Test Set: {len(test_df)} Images\n")
    
    # 3. BULLETPROOF CLASS WEIGHTING 
    def get_safe_weights(labels_array, num_classes):
        counts = np.bincount(labels_array, minlength=num_classes)
        weights = np.zeros(num_classes, dtype=np.float32)
        total_samples = len(labels_array)
        for i, count in enumerate(counts):
            if count > 0:
                weights[i] = total_samples / (num_classes * count)
            else:
                weights[i] = 0.0
        return weights

    body_weights = get_safe_weights(train_df['BodyPart_Encoded'].values, len(le_body.classes_))
    plane_weights = get_safe_weights(train_df['Plane_Encoded'].values, len(le_plane.classes_))
    
    # 🍏 APPLE SILICON HARDWARE ACTIVATION
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Apple M1 Max GPU detected! Using Metal Performance Shaders.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("⚠️ Warning: No GPU found. Using CPU.")

    tensor_body_weights = torch.tensor(body_weights, dtype=torch.float32).to(device)
    tensor_plane_weights = torch.tensor(plane_weights, dtype=torch.float32).to(device)

    # 4. DATA AUGMENTATION & LOADERS
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Bumping batch_size to 64 to feed the 32-core GPU faster!
    train_loader = DataLoader(UniverseDataset(train_df, "universe_ai_ready_images", train_transform), batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(UniverseDataset(val_df, "universe_ai_ready_images", val_test_transform), batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(UniverseDataset(test_df, "universe_ai_ready_images", val_test_transform), batch_size=64, shuffle=False, num_workers=4)

    # 5. INITIALIZE AI
    model = UniverseMedicalAI(len(le_mod.classes_), len(le_body.classes_), len(le_plane.classes_)).to(device)
    
    criterion_mod = nn.CrossEntropyLoss()
    criterion_body = nn.CrossEntropyLoss(weight=tensor_body_weights) 
    criterion_plane = nn.CrossEntropyLoss(weight=tensor_plane_weights)
    criterion_reg = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    epochs = 20
    best_val_loss = float('inf')
    
    print(f"🤖 Starting Training on {device}...\n")
    print(f"{'Epoch':<6} | {'Val Loss':<10} | {'Modality':<10} | {'BodyPart':<10} | {'Plane':<10}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        for images, mod, body, plane, space in train_loader:
            images, mod, body, plane, space = images.to(device), mod.to(device), body.to(device), plane.to(device), space.to(device)
            
            optimizer.zero_grad()
            pred_mod, pred_body, pred_plane, pred_space = model(images)
            
            loss_mod = criterion_mod(pred_mod, mod)
            loss_body = criterion_body(pred_body, body)
            loss_plane = criterion_plane(pred_plane, plane)
            
            valid_mask = ~torch.isnan(space[:, 0])
            loss_space = criterion_reg(pred_space[valid_mask], space[valid_mask]) if valid_mask.sum() > 0 else torch.tensor(0.0).to(device)
            
            total_loss = loss_mod + loss_body + loss_plane + (0.5 * loss_space)
            total_loss.backward()
            optimizer.step()
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, corr_mod, corr_body, corr_plane, total_samples = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            for images, mod, body, plane, space in val_loader:
                images, mod, body, plane, space = images.to(device), mod.to(device), body.to(device), plane.to(device), space.to(device)
                
                pred_mod, pred_body, pred_plane, pred_space = model(images)
                
                val_loss += (criterion_mod(pred_mod, mod) + criterion_body(pred_body, body) + criterion_plane(pred_plane, plane)).item()
                
                corr_mod += (torch.argmax(pred_mod, dim=1) == mod).sum().item()
                corr_body += (torch.argmax(pred_body, dim=1) == body).sum().item()
                corr_plane += (torch.argmax(pred_plane, dim=1) == plane).sum().item()
                total_samples += mod.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        acc_mod = (corr_mod / total_samples) * 100
        acc_body = (corr_body / total_samples) * 100
        acc_plane = (corr_plane / total_samples) * 100
        
        print(f"[{epoch+1:02d}/{epochs:02d}] | {avg_val_loss:<10.4f} | {acc_mod:<9.2f}% | {acc_body:<9.2f}% | {acc_plane:<9.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "universe_4task_ai.pth")
            
    # ==========================================
    # THE FINAL EXAM (TEST SET EVALUATION)
    # ==========================================
    print("\n" + "="*50)
    print("🎓 INITIATING FINAL EXAM ON SECRET TEST SET")
    print("="*50)
    
    model.load_state_dict(torch.load("universe_4task_ai.pth", weights_only=True))
    model.eval()
    
    test_corr_mod, test_corr_body, test_corr_plane, test_total = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, mod, body, plane, space in test_loader:
            images, mod, body, plane = images.to(device), mod.to(device), body.to(device), plane.to(device)
            pred_mod, pred_body, pred_plane, _ = model(images)
            
            test_corr_mod += (torch.argmax(pred_mod, dim=1) == mod).sum().item()
            test_corr_body += (torch.argmax(pred_body, dim=1) == body).sum().item()
            test_corr_plane += (torch.argmax(pred_plane, dim=1) == plane).sum().item()
            test_total += mod.size(0)

    print(f"\n✅ Final Exam Accuracy:")
    print(f"  Modality  : {(test_corr_mod / test_total) * 100:.2f}%")
    print(f"  Body Part : {(test_corr_body / test_total) * 100:.2f}%")
    print(f"  Image Plane: {(test_corr_plane / test_total) * 100:.2f}%\n")
    print("The Universe Foundation Model is complete and saved as 'universe_4task_ai.pth'.")

if __name__ == "__main__":
    train_model()