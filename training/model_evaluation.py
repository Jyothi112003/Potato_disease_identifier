from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_dataset, class_names):
    """
    Evaluate the model using F1 score and other metrics
    
    Args:
        model: Trained tensorflow model
        test_dataset: Test dataset
        class_names: List of class names
    """
    # Lists to store true labels and predictions
    y_true = []
    y_pred = []
    
    # Iterate through the test dataset
    for images, labels in test_dataset:
        # Get predictions
        predictions = model.predict(images)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Append true and predicted labels
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)
    
    # Calculate metrics
    # F1 score for each class
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Overall F1 score
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\nF1 Scores for each class:")
    for class_name, f1_value in zip(class_names, f1):
        print(f"{class_name}: {f1_value:.4f}")
    
    print(f"\nWeighted Average F1 Score: {f1_weighted:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Example usage:
"""
# Your class names for potato disease classification
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Evaluate your model
evaluate_model(your_trained_model, test_dataset, class_names)
"""
