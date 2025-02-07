import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Constants
IMAGE_SIZE = 256

class_names = ['Early Blight', 'Late Blight', 'Healthy']

def load_and_prep_image(image_path):
    """
    Reads an image from filepath, preprocesses it and reshapes it for the model
    """
    # Read the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array

def predict_image(model_path, image_path):
    """
    Predicts the class of a given image using the saved model
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = load_and_prep_image(image_path)
    
    # Make prediction
    pred = model.predict(img)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0]) * 100
    
    return {
        'class': pred_class,
        'confidence': confidence,
        'predictions': {class_names[i]: float(pred[0][i]) * 100 for i in range(len(class_names))}
    }

if __name__ == "__main__":
    # Example usage
    model_path = "potato_disease_model.keras"  # or "potato_disease_model.h5" if you used that format
    
    # Get image path from user
    image_path = input("Enter the path to your image: ")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    else:
        try:
            result = predict_image(model_path, image_path)
            print("\nPrediction Results:")
            print(f"Predicted Class: {result['class']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print("\nDetailed Predictions:")
            for class_name, probability in result['predictions'].items():
                print(f"{class_name}: {probability:.2f}%")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
