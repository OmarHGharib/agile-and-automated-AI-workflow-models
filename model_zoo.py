import torch
from monai.networks.nets import UNet, VNet, SwinUNETR

class MedicalModelZoo:
    def __init__(self, image_channels=4, num_classes=2, image_size=(128, 128, 64)):
        """
        image_channels: 4 (If you stacked T1, T1Gd, T2, FLAIR) or 1 (If just T1)
        num_classes: 2 (Background = 0, Tumor = 1)
        image_size: The standardized 3D size coming out of your Resampler
        """
        self.in_channels = image_channels
        self.out_classes = num_classes
        self.roi_size = image_size

    def get_model(self, model_name="unet3d"):
        """
        The Factory Method. Pass a string to get the full PyTorch model.
        """
        model_name = model_name.lower()

        if model_name == "unet3d":
            print(f"Loading Model: 3D U-Net ({self.in_channels} IN -> {self.out_classes} OUT)")
            return UNet(
                spatial_dims=3, # 3D data
                in_channels=self.in_channels,
                out_channels=self.out_classes,
                channels=(16, 32, 64, 128, 256), # The depth of the network
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )

        elif model_name == "vnet":
            print(f"Loading Model: V-Net ({self.in_channels} IN -> {self.out_classes} OUT)")
            return VNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.out_classes,
                dropout_prob=0.1
            )

        elif model_name == "swin_unetr":
            print(f"Loading Model: Swin-UNETR (Transformer) ({self.in_channels} IN -> {self.out_classes} OUT)")
            return SwinUNETR(
                img_size=self.roi_size, # Transformers need to know exact image size
                in_channels=self.in_channels,
                out_channels=self.out_classes,
                feature_size=48,
                use_checkpoint=True
            )

        else:
            raise ValueError(f"Model '{model_name}' is not in the zoo! Choose: unet3d, vnet, or swin_unetr")

# --- USAGE TEST ---
if __name__ == "__main__":
    # 1. Initialize the Zoo
    # Let's assume we are using 4 MRI sequences stacked together
    zoo = MedicalModelZoo(image_channels=4, num_classes=2, image_size=(128, 128, 64))

    # 2. Select your model dynamically
    selected_model_name = "vnet" # Try changing this to "swin_unetr" or "unet3d"
    
    my_ai = zoo.get_model(selected_model_name)

    # 3. Test it with a fake 3D brain tensor
    # Shape: (Batch_Size, Channels, X, Y, Z) -> (1, 4, 128, 128, 64)
    fake_patient_data = torch.randn(1, 4, 128, 128, 64)
    
    # Run the AI
    prediction_mask = my_ai(fake_patient_data)
    
    # The output should have 2 channels (Background vs Tumor)
    print(f"\nSuccess! Output Mask Shape: {prediction_mask.shape}")