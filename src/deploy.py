import argparse
import tensorflow as tf

def deploy_model(model_path):
    model = tf.keras.models.load_model(model_path)
    # Add deployment logic here, e.g., saving to a cloud service, etc.
    print(f"Model deployed from {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    deploy_model(args.model)