from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchinfo import summary
from torchvision import datasets
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np

import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageClassifier(nn.Module):
  def __init__(self):
      super().__init__()
      
      self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Second Conv Block
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Third Conv Block
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )
      self.fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 28 * 28, 512),  # Adjuast based on input image size (224x224)
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),  # Output for binary classification
        nn.Sigmoid()
        )

  def forward(self, x):
      x = self.conv_layers(x)
      x = self.fc_layers(x)
      return x
  
def transform_image(image):
    """
    This function converts the input PIL image to a NumPy array,
    applies a Gaussian filter to reduce noise, converts the image 
    back to a PIL image, and then applies further resizing, tensor 
    conversion, and normalization.
    """
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)
    
    # Apply Gaussian filter to reduce noise
    image_np = gaussian_filter(image_np, sigma=1)
    
    # Convert the NumPy array back to a PIL Image
    # Note: If image_np has type different from uint8, you might need to convert or clip values appropriately.
    image_pil = Image.fromarray(image_np.astype('uint8'))
    
    # Define the remaining transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Apply the defined transformations
    image = transform(image_pil)
    return image


def load_data(root = './data' ,batch_size = 16):
    
    mlflow.log_artifact(root, artifact_path="data")
    dataset = datasets.ImageFolder(root = root, transform=transform_image)
    # Example
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def training(model, train_loader, val_loader, epochs=10, lr=0.001):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # log params
    mlflow.log_params({
        "epochs":  epochs,
        "learning_rate": lr,
        "batch_size": 16,
        "optimizer": "adam",
        "loss_function": "cross_entropy",
        "metrics": "accuracy",
    })
    
    # Prepare input example and infer signature once, before training
    example_input, _ = next(iter(train_loader))
    example_input = example_input.to(device)
    example_output = model(example_input)
    signature = mlflow.models.infer_signature(example_input.cpu().detach().numpy(), example_output.cpu().detach().numpy())

    best_accuracy = 0.0
    name_epoch = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log training loss
        average_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")
        mlflow.log_metric(f"train_loss_epoch{epoch + 1}", average_loss, step=epoch)
        
        # Log the model with input example and signature to suppress warning
        mlflow.pytorch.log_model(
            model, 
            artifact_path=f"model_checkpoint_epoch{epoch + 1}",
            input_example= example_input.cpu().detach().numpy(), 
            signature = signature,
        )
        
        # Validation
        model.eval()
        val_loss = 0.0
        count_predicted = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                count_predicted += (predicted == labels).sum().item()
            accuracy_score = count_predicted / len(val_loader.dataset)
            
            if accuracy_score > best_accuracy:
                best_accuracy = accuracy_score
                name_epoch = epoch + 1
            mlflow.log_metric(f"val_accuracy_{epoch + 1}", accuracy_score, step=epoch)
            print(f"Validation Accuracy {epoch + 1}: {accuracy_score:.4f}")
            mlflow.log_metric(f"val_loss_{epoch+1}", val_loss / len(val_loader), step=epoch)
    mlflow.log_metric(f"Best_Val_acc_epoch_{name_epoch}", best_accuracy)
    return


def evaluation(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        accuracy_score = correct / len(test_loader.dataset)
        
        mlflow.log_metric("eval_accuracy", accuracy_score)
        print(f"Test Accuracy: {accuracy_score:.4f}")
    return accuracy_score


if __name__ == "__main__":
    mlflow.set_experiment("Lab01_tracking_experiment")
    mlflow.set_tracking_uri("http://0.0.0.0:5003")
    lr_list = [0.001, 0.01, 0.1]
    for lr in lr_list:
        with mlflow.start_run(run_name=f"Tune_with_lr_{lr}"):
            train_loader, val_loader, test_loader = load_data()
            model = ImageClassifier().to(device)
            training(model, train_loader, val_loader, epochs=10, lr=lr)

    
    