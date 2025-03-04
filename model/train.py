import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def create_model(input_shape=(64, 64, 3), num_classes=3):
    """Create a CNN model for sperm classification."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(data_dir, output_dir, batch_size=32, epochs=50, img_size=(64, 64)):
    """Train the model with the provided dataset."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Prepare training dataset
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Prepare validation dataset
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get class names and indices
    classes = list(train_generator.class_indices.keys())
    print(f"Classes: {classes}")
    
    # Create model
    model = create_model(input_shape=(*img_size, 3), num_classes=len(classes))
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Save class mapping
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for i, class_name in enumerate(classes):
            f.write(f"{i},{class_name}\n")
    
    print(f"Model training completed. Best model saved to {output_dir}/model.h5")
    return history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a sperm classification model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for training (square).')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        args.data_dir,
        args.output_dir,
        args.batch_size,
        args.epochs,
        (args.img_size, args.img_size)
    )