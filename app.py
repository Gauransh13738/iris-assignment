'''
This is a simple API that uses a pre-trained model to classify iris flowers into three classes: setosa, versicolor, and virginica.

This app shows the walkthrough build of deploying a simple ML application on a server to recieve requests using FastAPI.

Going forward , this can be containerized using Docker and deployed on the AWS Cloud using an S3 bucket or launching it on an EC2 instance.
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

app = FastAPI()

#Loading the model into our app
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Class map to map the flower names to numbers in the dataset
class_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Request schema - The inputs the app will recieve and feed to the model for output
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize API - This is the main app that will be used to run the server
app = FastAPI(title="Iris Classifier API")

# Root endpoint - Just for showing that the app is running
@app.get("/")
def root():
    return {"message": "Iris Classifier API is live!"}

# Predict endpoint - This is the endpoint that will be used to predict the class of the flower
# It takes in the input data and returns the predicted class
@app.post("/predict")
def predict(data: IrisInput):
    try:
        #Takes in feature inputs and converts them into a numpy array to be feeded into the model
        features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        prediction = model.predict(features)[0]
        return {"prediction": class_map[prediction]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Runs the app on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)