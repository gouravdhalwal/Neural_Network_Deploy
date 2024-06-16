import numpy as np
from src.config import config
from src.preprocessing.data_management import load_model

def load_trained_model():
    model = load_model("two_input_xor_nn.pkl")
    theta0 = model["params"]["biases"]
    theta = model["params"]["weights"]
    activations = model["activations"]
    return theta0, theta, activations

def activation_function(z, activation):
    if activation == "linear":
        return z
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif activation == "tanh":
        return np.tanh(z)
    elif activation == "relu":
        return np.maximum(0, z)
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def forward_pass(X, theta0, theta, activations):
    h = X.reshape(1, -1)  # Reshape input for single prediction
    for l in range(1, len(theta0)):
        z = np.dot(h, theta[l]) + theta0[l]
        h = activation_function(z, activations[l])
    return h

def predict(input_data):
    theta0, theta, activations = load_trained_model()
    predictions = []
    for input_sample in input_data:
        prediction = forward_pass(input_sample, theta0, theta, activations)
        predictions.append(prediction)
    return predictions

def calculate_accuracy(predictions, expected_outputs):
    correct_count = 0
    for i, prediction in enumerate(predictions):
        predicted_class = 1 if prediction[0][0] >= 0.5 else 0
        if predicted_class == expected_outputs[i]:
            correct_count += 1
    accuracy = correct_count / len(expected_outputs) * 100
    return accuracy

if __name__ == "__main__":
    input_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    expected_outputs = np.array([0, 1, 1, 0])  # Expected outputs for XOR function

    predictions = predict(input_data)
    accuracy = calculate_accuracy(predictions, expected_outputs)

    for i, pred in enumerate(predictions):
        print(f"Prediction for input {input_data[i]}: {pred[0][0]}")

    print(f"Accuracy: {accuracy:.2f}%")
