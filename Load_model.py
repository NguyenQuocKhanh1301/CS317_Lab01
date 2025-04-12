from torchvision import transforms, models
import torch
import mlflow
from mlflow.models import Model
import cv2 as cv
from lab01 import *
model_version = 1
model_name = "best_model"
model_uri = f"models:/{model_name}/{model_version}"

loaded_model = mlflow.pyfunc.load_model(model_uri)

image_path = './data/cats_set_0/cat.4001.jpg'
image = Image.open(image_path)
# Preprocess the image
input_tensor = transform_image(image)        # Shape becomes (3, 224, 224)
# Add a batch dimension: (1, 3, 224, 224)
input_batch = input_tensor.unsqueeze(0)

# image = transform_image(image)
output = loaded_model.predict(input_batch.numpy())
pred = torch.argmax(torch.tensor(output), dim=1).item()
print(f"Predicted class: {pred}")
