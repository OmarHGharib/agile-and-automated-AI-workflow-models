import tensorflow as tf
from tensorflow.keras import layers, models, Model

def build_metadata_imputer(input_shape=(256, 256, 1)):
    # 1. The Backbone (Feature Extractor)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x) # Flatten features

    # 2. Head A: Modality (Classification)
    # Output: [CT, MRI, XRay]
    modality_out = layers.Dense(3, activation='softmax', name='modality')(x)

    # 3. Head B: Body Part (Classification)
    # Output: [Head, Chest, Abdomen, Pelvis]
    bodypart_out = layers.Dense(4, activation='softmax', name='body_part')(x)

    # 4. Head C: Pixel Spacing (Regression)
    # Output: Single float value (mm per pixel)
    spacing_out = layers.Dense(1, activation='linear', name='pixel_spacing')(x)

    # Combine into one model
    model = Model(inputs=inputs, outputs=[modality_out, bodypart_out, spacing_out])
    
    return model

# --- Compile the Model ---
model = build_metadata_imputer()
model.compile(
    optimizer='adam',
    loss={
        'modality': 'categorical_crossentropy',
        'body_part': 'categorical_crossentropy',
        'pixel_spacing': 'mse' # Mean Squared Error for regression
    },
    loss_weights={
        'modality': 1.0,
        'body_part': 1.0,
        'pixel_spacing': 0.1 # Weight regression less heavily
    }
)

model.summary()