import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# --- Import Your Custom Stations ---
from dataset import CancerDataset
from model_zoo import MedicalModelZoo
from loss import DiceLoss

def train_model():
    print("=== STARTING THE AI FACTORY ===")

    # 1. Setup Hyperparameters (The Control Dials)
    EPOCHS = 50
    # 3D medical data is massive. If your computer crashes with an "Out of Memory" error, change BATCH_SIZE to 1.
    BATCH_SIZE = 2  
    LEARNING_RATE = 1e-4
    
    # Automatically use the Graphic Card (GPU) if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on Device: {DEVICE}")

    # 2. Load the Data (The Conveyor Belt)
    # Using the exact files you found in your folder!
    train_patients = [
        {
            "image": "/Users/omargharib/Desktop/A stage/001/t0/1_t0_t1gd.nii.gz",
            "mask": "/Users/omargharib/Desktop/A stage/001/t0/1_t0_gtv.nii.gz"
        }
        # You will add Patient 002, Patient 003, etc., here later.
    ]
    
    print("[*] Initializing Data Pipeline...")
    train_dataset = CancerDataset(train_patients, ai_weights="ultimate_4task_ai.pth")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize the AI (The Brain)
    # in_channels=1 (Just the T1Gd MRI)
    # out_classes=1 (Just the Tumor Mask)
    zoo = MedicalModelZoo(image_channels=1, num_classes=1, image_size=(128, 128, 64))
    model = zoo.get_model("unet3d").to(DEVICE)

    # 4. Initialize the Teacher & Optimizer
    teacher = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    # --- THE TRAINING LOOP ---
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train() # Put model in learning mode
        epoch_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        for batch_idx, batch in enumerate(train_loader):
            # A. Move data to GPU
            images = batch["image"].to(DEVICE)
            true_masks = batch["mask"].to(DEVICE) # The Doctor's GTV Mask!

            # B. Clear old memory from the previous step
            optimizer.zero_grad()

            # C. The AI makes a guess (Forward Pass)
            predicted_masks = model(images)

            # D. The Teacher grades the guess using Dice Math
            loss = teacher(predicted_masks, true_masks)

            # E. The AI learns from its mistakes (Backpropagation)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"   Batch {batch_idx+1}/{len(train_loader)} | Dice Loss: {loss.item():.4f}")

        # --- END OF EPOCH LOGIC ---
        avg_loss = epoch_loss / len(train_loader)
        print(f"[*] Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

        # Save the model if it's the best one we've seen so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_tumor_model.pth")
            print(f"[*] New High Score! Model saved to 'best_tumor_model.pth'")

if __name__ == "__main__":
    train_model()