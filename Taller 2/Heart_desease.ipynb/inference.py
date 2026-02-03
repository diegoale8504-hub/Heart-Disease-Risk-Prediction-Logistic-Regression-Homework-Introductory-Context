import numpy as np
import json

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    w = np.load(f"{model_dir}/weights.npy")
    b = np.load(f"{model_dir}/bias.npy")
    return (w, b)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["inputs"])
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    w, b = model
    z = np.dot(input_data, w) + b
    probs = sigmoid(z)
    return probs

def output_fn(prediction, content_type):
    return json.dumps({"probability": prediction.tolist()})
