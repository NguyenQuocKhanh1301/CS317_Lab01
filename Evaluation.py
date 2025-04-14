from torchvision import transforms, models
import torch
import mlflow
from mlflow.models import Model
import cv2 as cv
from lab01 import *

mlflow.set_experiment("Lab01_tracking_experiment")
mlflow.set_tracking_uri("http://0.0.0.0:5003")
model_version = 1
model_name = "best_model"
model_uri = f"models:/{model_name}/{model_version}"

loaded_model = mlflow.pytorch.load_model(model_uri).to(device)
with mlflow.start_run(run_name=f"Evaluation_with_{model_name}"):
    # Load the test data
    test_loader = load_data()[2]
    evaluation(loaded_model, test_loader)
