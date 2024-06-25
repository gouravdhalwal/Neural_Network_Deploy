from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from src.preprocessing.data_management import load_model
from src.predict import predict  # Correct import

# Load the trained model
saved_file_name = "two_input_xor_nn.pkl"
loaded_model = load_model(saved_file_name)

app = FastAPI(
    title="Two Input XOR Function Implementor",
    description="A two input Neural Network to implement XOR Function",
    version="0.1"
)

# CORS middleware setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TwoInputXorGate(BaseModel):
    X1: float
    X2: float

@app.get("/")
def index():
    return {"message": "A Web App for serving the output of a two input XOR function implemented through neural network"}

@app.post("/GenerateResponse")
def generate_response(trigger: TwoInputXorGate):
    input1 = trigger.X1
    input2 = trigger.X2

    input_to_nn = np.array([[input1, input2]])
    predictions = predict(input_to_nn)  # Use the predict function directly
    nn_out = predictions[0]  # Get the prediction for the single input

    return {"prediction": nn_out.tolist()}  # Ensure the output is JSON serializable

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
