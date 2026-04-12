import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
ORIGINAL_IMAGE_PATH = "/path/to/your/image.jpg"  # Original harvest image (from COCO dataset)
JSON_MASK_PATH = "/path/to/your/dataset.json"   # COCO format JSON file
OUTPUT_MODEL_PATH = "/home/claude/potato_model.h5"
IMAGE_ID = 0  # Which image from the JSON to use (0-indexed)
IMAGE_HEIGHT = 360  # From COCO JSON
IMAGE_WIDTH = 640   # From COCO JSON
NUM_CLASSES = 5  # stick, Stone, big stone, potato-detection, background

# Class mapping - from COCO JSON categories (id -> name)
CLASS_MAPPING = {
    0: "potato-detection",  # category_id 0
    1: "Stone",             # category_id 1
    2: "big stone",         # category_id 2
    3: "stick"              # category_id 3
}

# Create reverse mapping for convenience
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAPPING.items()}
CLASS_NAME_TO_ID["background"] = 4  # Add background class

# ============================================================================
# STEP 1: LOAD AND PARSE JSON MASK
# ============================================================================
def decode_rle(rle_data, image_size):
    """Decode RLE (Run-Length Encoding) to binary mask"""
    counts = rle_data['counts']
    h, w = image_size
    
    if isinstance(counts, str):
        # String format - decode it
        mask = np.zeros(h * w, dtype=np.uint8)
        current_pos = 0
        current_value = 0
        
        i = 0
        while i < len(counts):
            # Decode run length
            run_length = 0
            power = 1
            while i < len(counts):
                char_val = ord(counts[i]) - 48
                if char_val > 26:
                    char_val -= 58
                run_length += char_val * power
                power *= 27
                i += 1
                if power > 27 ** 2:
                    break
            
            # Set pixels
            mask[current_pos:current_pos + run_length] = current_value
            current_pos += run_length
            current_value = 1 - current_value
        
        return mask.reshape((h, w))
    else:
        # List format
        mask = np.zeros(h * w, dtype=np.uint8)
        current_pos = 0
        current_value = 0
        for run_length in counts:
            mask[current_pos:current_pos + run_length] = current_value
            current_pos += run_length
            current_value = 1 - current_value
        return mask.reshape((h, w))


def load_mask_from_json(json_path, image_id, image_shape):
    """
    Load mask from COCO JSON format for a specific image.
    Creates a multi-class mask where each pixel has a class ID.
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    h, w = image_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 4  # Initialize with background (4)
    
    # Get all annotations for this image
    for annotation in coco_data['annotations']:
        if annotation['image_id'] != image_id:
            continue
        
        category_id = annotation['category_id']
        
        # Decode RLE segmentation
        if 'segmentation' in annotation and 'counts' in annotation['segmentation']:
            seg_mask = decode_rle(annotation['segmentation'], (h, w))
            # Apply this object's mask
            mask[seg_mask == 1] = category_id
        
        # Fallback: use polygon if available (for some annotations)
        elif 'segmentation' in annotation and isinstance(annotation['segmentation'], list):
            if len(annotation['segmentation']) > 0:
                points = np.array(annotation['segmentation'][0], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], category_id)
    
    return mask


# ============================================================================
# STEP 2: DATA AUGMENTATION (to create training set from 1 image)
# ============================================================================
def augment_image(image, mask, num_augmentations=20):
    """Generate augmented versions of the image and mask"""
    augmented_images = [image]
    augmented_masks = [mask]
    
    for _ in range(num_augmentations):
        # Random rotation
        angle = np.random.randint(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, rot_matrix, (w, h))
        rotated_mask = cv2.warpAffine(mask, rot_matrix, (w, h))
        
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        bright_img = np.clip(rotated_img * brightness, 0, 255).astype(np.uint8)
        
        # Random flip
        if np.random.rand() > 0.5:
            bright_img = cv2.flip(bright_img, 1)
            rotated_mask = cv2.flip(rotated_mask, 1)
        
        augmented_images.append(bright_img)
        augmented_masks.append(rotated_mask)
    
    return augmented_images, augmented_masks


# ============================================================================
# STEP 3: BUILD SEGMENTATION MODEL (Keras)
# ============================================================================
def build_segmentation_model(height, width, num_classes):
    """Build a simple U-Net style segmentation model"""
    
    inputs = layers.Input(shape=(height, width, 3))
    
    # Encoder (downsampling)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder (upsampling)
    u3 = layers.UpSampling2D((2, 2))(c4)
    u3 = layers.concatenate([u3, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    
    u1 = layers.UpSampling2D((2, 2))(c6)
    u1 = layers.concatenate([u1, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# ============================================================================
# STEP 4: PREPARE DATA
# ============================================================================
def prepare_data(original_image_path, json_mask_path, image_id=0, num_augmentations=20):
    """Load image, mask, and augment"""
    
    # Load original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Load mask from JSON for this specific image
    mask = load_mask_from_json(json_mask_path, image_id, original_image.shape)
    
    print(f"Loaded image {image_id} with mask shape {mask.shape}")
    print(f"Unique classes in mask: {np.unique(mask)}")
    
    # Augment
    aug_images, aug_masks = augment_image(original_image, mask, num_augmentations)
    
    # Convert to arrays
    X = np.array(aug_images, dtype=np.float32) / 255.0
    y = np.array(aug_masks, dtype=np.int32)
    
    # Convert masks to one-hot encoding
    y_onehot = np.zeros((y.shape[0], y.shape[1], y.shape[2], NUM_CLASSES), dtype=np.float32)
    for i in range(NUM_CLASSES):
        y_onehot[:, :, :, i] = (y == i).astype(np.float32)
    
    return X, y_onehot, y


# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
def train_model(X, y_onehot):
    """Build and train the model"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    # Build model
    model = build_segmentation_model(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=4,
        verbose=1
    )
    
    return model, history


# ============================================================================
# STEP 6: CALCULATE RATIOS
# ============================================================================
def calculate_ratios(mask_pred, exclude_background=True):
    """
    Calculate class ratios and potato vs clutter ratio
    
    exclude_background: if True, ratios exclude background pixels
    """
    
    # Flatten mask
    flat_mask = mask_pred.flatten()
    unique, counts = np.unique(flat_mask, return_counts=True)
    
    total_pixels = len(flat_mask)
    
    print("\n" + "="*60)
    print("SEGMENTATION RESULTS")
    print("="*60)
    
    # Calculate ratios
    ratios = {}
    for class_id, count in zip(unique, counts):
        class_name = CLASS_MAPPING.get(class_id, "background")
        ratio = count / total_pixels
        ratios[class_name] = ratio
        print(f"{class_name.upper():20} | Pixels: {count:8} | Ratio: {ratio*100:6.2f}%")
    
    # Exclude background if requested
    if exclude_background:
        bg_pixels = ratios.get("background", 0) * total_pixels
        total_non_bg = total_pixels - bg_pixels
        print("\n--- Excluding Background ---")
        for class_name, ratio in ratios.items():
            if class_name != "background":
                adjusted_ratio = (ratio * total_pixels) / total_non_bg if total_non_bg > 0 else 0
                print(f"{class_name.upper():20} | Adjusted Ratio: {adjusted_ratio*100:6.2f}%")
    
    # Potato vs Clutter ratio
    potato_pixels = ratios.get("potato-detection", 0) * total_pixels
    clutter_pixels = (ratios.get("stick", 0) + ratios.get("Stone", 0) + ratios.get("big stone", 0)) * total_pixels
    
    if clutter_pixels > 0:
        potato_clutter_ratio = potato_pixels / clutter_pixels
        print(f"\n--- Potato vs Clutter Ratio ---")
        print(f"Potato Pixels:  {int(potato_pixels)}")
        print(f"Clutter Pixels: {int(clutter_pixels)}")
        print(f"Ratio (Potato/Clutter): {potato_clutter_ratio:.2f}")
    else:
        print(f"\n--- Potato vs Clutter Ratio ---")
        print(f"No clutter detected or insufficient data")
    
    print("="*60 + "\n")
    
    return ratios


# ============================================================================
# STEP 7: MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    print("Starting Potato Segmentation Pipeline...")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {list(CLASS_MAPPING.values())}\n")
    
    # Step 1: Load and augment data
    print("Step 1: Loading image and mask from COCO JSON...")
    X, y_onehot, y_original = prepare_data(ORIGINAL_IMAGE_PATH, JSON_MASK_PATH, IMAGE_ID, num_augmentations=20)
    print(f"Generated {len(X)} training samples through augmentation\n")
    
    # Step 2: Train model
    print("Step 2: Training model...")
    model, history = train_model(X, y_onehot)
    
    # Step 3: Save model
    print(f"\nStep 3: Saving model to {OUTPUT_MODEL_PATH}...")
    model.save(OUTPUT_MODEL_PATH)
    print("Model saved!\n")
    
    # Step 4: Make prediction on original image
    print("Step 4: Making prediction on original image...")
    original_image = cv2.imread(ORIGINAL_IMAGE_PATH)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_normalized = original_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_img = np.expand_dims(original_image_normalized, axis=0)
    
    # Predict
    prediction = model.predict(input_img)
    predicted_mask = np.argmax(prediction[0], axis=-1)
    
    # Step 5: Calculate ratios
    print("Step 5: Calculating class ratios...")
    ratios = calculate_ratios(predicted_mask, exclude_background=True)
    
    # Step 6: Visualize results
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Predicted mask (colored)
    mask_colored = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    colors = {
        0: [255, 0, 0],      # stick - red
        1: [0, 255, 0],      # Stone - green
        2: [0, 0, 255],      # big stone - blue
        3: [255, 255, 0],    # potato-detection - yellow
        4: [128, 128, 128]   # background - gray
    }
    for class_id, color in colors.items():
        mask_colored[predicted_mask == class_id] = color
    
    axes[1].imshow(mask_colored)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis('off')
    
    # Class distribution
    class_names_list = [CLASS_MAPPING.get(i, "background") for i in range(NUM_CLASSES)]
    class_counts = [np.sum(predicted_mask == i) for i in range(NUM_CLASSES)]
    
    axes[2].bar(class_names_list, class_counts)
    axes[2].set_title("Pixel Count by Class")
    axes[2].set_ylabel("Pixel Count")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/claude/segmentation_results.png', dpi=100, bbox_inches='tight')
    print("Visualization saved to /home/claude/segmentation_results.png\n")
    
    print("Pipeline complete!")