import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        smooth: A tiny number added to the math to prevent dividing by zero
        if the AI predicts a completely blank image.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: The raw output from your 3D U-Net (Logits).
        targets: The ground truth mask drawn by the doctor (0=Healthy, 1=Tumor).
        """
        # 1. Squash the AI's raw numbers into probabilities between 0.0 and 1.0
        inputs = torch.sigmoid(inputs)
        
        # 2. Flatten the 3D brain tensors into long 1D mathematical arrays
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 3. Calculate the Overlap (Intersection)
        intersection = (inputs * targets).sum()
        
        # 4. Calculate the Dice Score formula
        # Dice = (2 * Overlap) / (Total AI Pixels + Total Doctor Pixels)
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 5. Return the Loss (1 - Score)
        return 1.0 - dice_score

# --- USAGE TEST ---
if __name__ == "__main__":
    teacher = DiceLoss()

    # Simulate a tiny 2x2 mask from a doctor (1 is tumor, 0 is background)
    doctor_mask = torch.tensor([[1.0, 1.0], 
                                [0.0, 0.0]])

    # Simulate a bad AI prediction (guessed the wrong side!)
    bad_ai_prediction = torch.tensor([[0.0, 0.0], 
                                      [1.0, 1.0]])
    
    # Simulate a perfect AI prediction
    good_ai_prediction = torch.tensor([[1.0, 1.0], 
                                       [0.0, 0.0]])

    # Calculate losses (Note: AI predictions usually need to be raw logits, 
    # but for this visual test, we pretend the sigmoid already squashed them to large positive/negative numbers)
    
    print(f"Bad AI Loss: {teacher(bad_ai_prediction * 10, doctor_mask):.4f} (Closer to 1.0 is bad)")
    print(f"Good AI Loss: {teacher(good_ai_prediction * 10, doctor_mask):.4f} (Closer to 0.0 is good)")