import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, LSTM, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os


class ConvLSTMModel:
    def __init__(self, input_shape, num_classes, dataset_path="Dataset/breathing_dataset.csv"):
        """
        Initialize the Conv-LSTM model for sleep stage classification.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            num_classes: Number of classes for classification
            dataset_path: Path to the dataset CSV file
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        
    def load_dataset(self):
        """
        Load the dataset from the specified path.
        
        Returns:
            DataFrame containing the dataset
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        print(f"Loading dataset from: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded successfully with {len(df)} samples")
        return df
    
    def get_dataset_info(self):
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
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
        """
        Create and compile the Conv-LSTM model.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First convolutional block
            Conv1D(64, 7, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # Second convolutional block
            Conv1D(128, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # LSTM layer with dropout for regularization
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            
            # Global average pooling to reduce dimensionality
            GlobalAveragePooling1D(),
            
            # Fully connected layers
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
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
        """
        Get model summary for debugging and visualization.
        
        Returns:
            Model summary
        """
        model = self.create_model()
        return model.summary()
    
    def get_recommended_epochs(self):
        """
        Get recommended number of epochs for training.
        
        Returns:
            Number of epochs
        """
        return 10
    
    def get_recommended_batch_size(self):
        """
        Get recommended batch size for training.
        
        Returns:
            Batch size
        """
        return 64


# Example usage
if __name__ == "__main__":
    # Example for breathing irregularity detection
    input_shape = (961, 3)  # 961 time points, 3 signals
    num_classes = 3  # Normal, Hypopnea, Obstructive Apnea
    
    # Create Conv-LSTM model with dataset path
    conv_lstm = ConvLSTMModel(input_shape, num_classes, dataset_path="Dataset/breathing_dataset.csv")
    
    # Get dataset information
    print("Dataset Information:")
    print("=" * 50)
    dataset_info = conv_lstm.get_dataset_info()
    if dataset_info:
        for key, value in dataset_info.items():
            print(f"{key}: {value}")
    
    # Create model
    model = conv_lstm.create_model()
    
    # Print model summary
    print("\nConv-LSTM Model Summary:")
    print("=" * 50)
    model.summary()
    
    # Print recommended training parameters
    print(f"\nRecommended training parameters:")
    print(f"Epochs: {conv_lstm.get_recommended_epochs()}")
    print(f"Batch size: {conv_lstm.get_recommended_batch_size()}")
    print(f"Dataset path: {conv_lstm.dataset_path}")