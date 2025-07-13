import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os


class CNNModel:
    def __init__(self, input_shape, num_classes, dataset_path="Dataset/breathing_dataset.csv"):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        
    def load_dataset(self):
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        print(f"Loading dataset from: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded successfully with {len(df)} samples")
        return df
    
    def get_dataset_info(self):
        
        try:
            df = self.load_dataset()
            info = {
                'total_samples': len(df),
                'participants': df['participant_id'].nunique() if 'participant_id' in df.columns else 'N/A',
                'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else 'N/A',
                'columns': df.columns.tolist()
            }
            return info
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return None
        
    def create_model(self):
        
        model = Sequential([
            # First convolutional block
            Conv1D(32, 11, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # Second convolutional block
            Conv1D(64, 9, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # Third convolutional block
            Conv1D(128, 7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # Fourth convolutional block
            Conv1D(256, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model_summary(self):
        
        model = self.create_model()
        return model.summary()
    
    def get_recommended_epochs(self):
        
        return 20
    
    def get_recommended_batch_size(self):
        
        return 64



if __name__ == "__main__":
    # Example for breathing irregularity detection
    input_shape = (961, 3)  # 961 time points, 3 signals
    num_classes = 3  # Normal, Hypopnea, Obstructive Apnea
    
    cnn = CNNModel(input_shape, num_classes, dataset_path="Dataset/breathing_dataset.csv")
    
    # Get dataset information
    print("Dataset Information:")
    print("=" * 50)
    dataset_info = cnn.get_dataset_info()
    if dataset_info:
        for key, value in dataset_info.items():
            print(f"{key}: {value}")
    
    # Create model
    model = cnn.create_model()
    
    # Print model summary
    print("\n1D CNN Model Summary:")
    print("=" * 50)
    model.summary()
    
    # Print recommended training parameters
    print(f"\nRecommended training parameters:")
    print(f"Epochs: {cnn.get_recommended_epochs()}")
    print(f"Batch size: {cnn.get_recommended_batch_size()}")
    print(f"Dataset path: {cnn.dataset_path}")