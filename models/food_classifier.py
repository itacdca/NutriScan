import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

class FoodClassifier:
    """Food classification model using transfer learning with MobileNetV2"""
    
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.num_classes = 101  # Food-101 dataset classes
        self.load_model()
    
    def create_model(self):
        """Create the food classification model using transfer learning"""
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model(self):
        """Load or create the model"""
        try:
            # Try to load existing model
            if os.path.exists('food_model.h5'):
                self.model = tf.keras.models.load_model('food_model.h5')
            else:
                # Create new model with pre-trained weights
                self.model = self.create_model()
                self._load_pretrained_weights()
        except Exception as e:
            # Fallback: create new model
            print(f"Warning: Could not load saved model ({e}). Creating new model.")
            self.model = self.create_model()
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pre-trained weights or use transfer learning initialization"""
        try:
            # This would typically load weights trained on Food-101 dataset
            # For now, we use the MobileNetV2 ImageNet weights + random classification head
            print("Using transfer learning from ImageNet weights")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
    
    def preprocess_input(self, image_array):
        """Preprocess input for the model"""
        # Normalize to [-1, 1] range (MobileNetV2 preprocessing)
        image_array = tf.cast(image_array, tf.float32)
        image_array = (image_array / 127.5) - 1.0
        return image_array
    
    def predict(self, processed_image):
        """Make prediction on processed image"""
        try:
            # Ensure image is in correct format
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Preprocess for model
            model_input = self.preprocess_input(processed_image)
            
            # Make prediction
            predictions = self.model.predict(model_input, verbose=0)
            
            # Get confidence (max probability)
            confidence = np.max(predictions[0])
            
            return predictions[0], confidence
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_model_info(self):
        """Get information about the model"""
        if self.model:
            return {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            }
        return None
