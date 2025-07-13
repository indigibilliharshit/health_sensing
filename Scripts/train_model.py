import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class BreathingClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        
    def create_1d_cnn(self):
        model = Sequential([
            Conv1D(32, 11, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Conv1D(64, 9, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Conv1D(128, 7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Conv1D(256, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_conv_lstm(self):
        model = Sequential([
            Conv1D(64, 7, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Conv1D(128, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # Reshape for LSTM - LSTM expects 3D input (samples, timesteps, features)
            tf.keras.layers.Reshape((-1, 128)),
            
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def prepare_data(self, df):
        # Extract signals and combine them
        signals = []
        for idx, row in df.iterrows():
            flow = np.array(eval(row['flow_signal']))
            thoracic = np.array(eval(row['thoracic_signal']))
            spo2 = np.array(eval(row['spo2_signal']))
            
            # Combine all three signals
            combined_signal = np.column_stack([flow, thoracic, spo2])
            signals.append(combined_signal)
        
        X = np.array(signals)
        y = df['label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)
        
        return X, y_categorical, y_encoded
    
    def calculate_metrics(self, y_true, y_pred, class_names):
        # Convert from categorical to class indices
        y_true_idx = np.argmax(y_true, axis=1)
        y_pred_idx = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_idx, y_pred_idx)
        precision, recall, f1, support = precision_recall_fscore_support(y_true_idx, y_pred_idx, average=None)
        
        # Calculate sensitivity and specificity for each class
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        sensitivity = []
        specificity = []
        
        for i in range(len(class_names)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivity.append(sens)
            specificity.append(spec)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm,
            'support': support
        }
    
    def train_and_evaluate(self, df, model_type='cnn'):
        participants = df['participant_id'].unique()
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*60}")
        
        for fold, test_participant in enumerate(participants):
            print(f"\nFold {fold + 1}/5 - Test Participant: {test_participant}")
            print("-" * 40)
            
            # Split data
            train_df = df[df['participant_id'] != test_participant]
            test_df = df[df['participant_id'] == test_participant]
            
            # Prepare data
            X_train, y_train, y_train_encoded = self.prepare_data(train_df)
            X_test, y_test, y_test_encoded = self.prepare_data(test_df)
            
            # Calculate class weights for imbalanced data
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_encoded),
                y=y_train_encoded
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Create model
            if model_type == 'cnn':
                model = self.create_1d_cnn()
                epochs = 20
            else:
                model = self.create_conv_lstm()
                epochs = 15
            
            # Callbacks
            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=64,
                validation_data=(X_test, y_test),
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Predict
            y_pred = model.predict(X_test, verbose=0)
            
            # Calculate metrics
            class_names = self.label_encoder.classes_
            metrics = self.calculate_metrics(y_test, y_pred, class_names)
            
            # Store results
            fold_results = {
                'fold': fold + 1,
                'test_participant': test_participant,
                'metrics': metrics,
                'class_names': class_names
            }
            all_results.append(fold_results)
            
            # Print fold results
            print(f"\nFold {fold + 1} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print("\nPer-class metrics:")
            for i, class_name in enumerate(class_names):
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision'][i]:.4f}")
                print(f"  Recall: {metrics['recall'][i]:.4f}")
                print(f"  Sensitivity: {metrics['sensitivity'][i]:.4f}")
                print(f"  Specificity: {metrics['specificity'][i]:.4f}")
        
        # Aggregate results
        self.print_aggregate_results(all_results, model_type)
        
        return all_results
    
    def print_aggregate_results(self, all_results, model_type):
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} Model - Aggregate Results")
        print(f"{'='*60}")
        
        # Collect all metrics
        accuracies = [result['metrics']['accuracy'] for result in all_results]
        
        class_names = all_results[0]['class_names']
        num_classes = len(class_names)
        
        precisions = np.array([result['metrics']['precision'] for result in all_results])
        recalls = np.array([result['metrics']['recall'] for result in all_results])
        sensitivities = np.array([result['metrics']['sensitivity'] for result in all_results])
        specificities = np.array([result['metrics']['specificity'] for result in all_results])
        
        # Print overall accuracy
        print(f"Overall Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        
        # Print per-class aggregate metrics
        print("\nPer-class Aggregate Metrics:")
        print("-" * 50)
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name}:")
            print(f"  Precision: {np.mean(precisions[:, i]):.4f} ± {np.std(precisions[:, i]):.4f}")
            print(f"  Recall: {np.mean(recalls[:, i]):.4f} ± {np.std(recalls[:, i]):.4f}")
            print(f"  Sensitivity: {np.mean(sensitivities[:, i]):.4f} ± {np.std(sensitivities[:, i]):.4f}")
            print(f"  Specificity: {np.mean(specificities[:, i]):.4f} ± {np.std(specificities[:, i]):.4f}")
        
        # Print aggregate confusion matrix
        print(f"\nAggregate Confusion Matrix:")
        print("-" * 30)
        total_cm = np.zeros((num_classes, num_classes))
        for result in all_results:
            total_cm += result['metrics']['confusion_matrix']
        
        print("Predicted ->")
        print("Actual |")
        header = "        " + " ".join(f"{name:>8}" for name in class_names)
        print(header)
        for i, class_name in enumerate(class_names):
            row = f"{class_name:>8} " + " ".join(f"{int(total_cm[i, j]):>8}" for j in range(num_classes))
            print(row)

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('Dataset/breathing_dataset.csv')
    
    print(f"Dataset loaded successfully")
    print(f"Total samples: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    # Initialize classifier
    input_shape = (961, 3)  # 961 time points, 3 signals
    num_classes = 3
    
    classifier = BreathingClassifier(input_shape, num_classes)
    
    # Train and evaluate both models
    print("\n" + "="*80)
    print("BREATHING IRREGULARITY DETECTION - MODEL TRAINING & EVALUATION")
    print("="*80)
    
    # Train 1D CNN
    cnn_results = classifier.train_and_evaluate(df, model_type='cnn')
    
    # Train Conv-LSTM
    lstm_results = classifier.train_and_evaluate(df, model_type='conv_lstm')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()