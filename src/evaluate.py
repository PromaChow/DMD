import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import argparse

def evaluate_model(model_path):
    data = np.load('data.npz')
    X_test, y_test = data['X_test'], data['y_test']
    
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Confusion Matrix:\n{conf_matrix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    evaluate_model(args.model)