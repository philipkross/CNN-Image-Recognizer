import torch
from torch import nn
import matplotlib.pyplot as plt
import random
import numpy as np
import requests
import zipfile
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import TrivialAugmentWide
import kagglehub, shutil, random, os
from pathlib import Path



# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Accuracy function
# -----------------------------
def accufn(ytrue, ypred):
    correct = torch.eq(ytrue, ypred).sum().item()
    return (correct / len(ypred)) * 100

# -----------------------------
# Training function
# -----------------------------
def training(model, dataloader, lossfn, optimizer, device, accufn, print_every=400):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = lossfn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        total_acc += accufn(y, preds)
        
        if batch_idx % print_every == 0:
            print(f"Looked at {batch_idx*len(X)}/{len(dataloader.dataset)} samples.")
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc

# -----------------------------
# Testing function
# -----------------------------
def testing(model, dataloader, lossfn, device, accufn):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = lossfn(logits, y)
            total_loss += loss.item()
            
            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            total_acc += accufn(y, preds)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc

# -----------------------------
# Dataset download & setup
# -----------------------------
datapath = Path("data/")
imagepath = datapath / "pizza_steak_sushi"

# # Setup pizza_steak_sushi dataset
# if not imagepath.is_dir():
#     imagepath.mkdir(parents=True, exist_ok=True)
#     with open(datapath / "pizza_steak_sushi.zip", "wb") as f:
#         request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#         f.write(request.content)
#     with zipfile.ZipFile(datapath / "pizza_steak_sushi.zip", "r") as zip_ref:
#         zip_ref.extractall(imagepath)

traindir = imagepath / "train"
testdir = imagepath / "test"

# # Helper: copy N random images
# def copy_subset(src, dst, n=1000):
#     files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
#     subset = random.sample(files, min(n, len(files)))
#     for fname in subset:
#         shutil.copy(src / fname, dst / fname)

# # Setup Stanford Cars dataset
# cache_path = Path(kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset"))
# train_cars_target = imagepath / "train" / "cars"
# test_cars_target = imagepath / "test" / "cars"
# train_cars_target.mkdir(parents=True, exist_ok=True)
# test_cars_target.mkdir(parents=True, exist_ok=True)

# # Only copy if folders are empty
# if not any(train_cars_target.iterdir()):
#     train_cars_source = cache_path / "cars_train" / "cars_train"
#     test_cars_source = cache_path / "cars_test" / "cars_test"
#     copy_subset(train_cars_source, train_cars_target, 1000)
#     copy_subset(test_cars_source, test_cars_target, 20)

# # Setup Bikes dataset from local zip
# bikes_zip_path = Path(r"C:\Users\kross\Documents\AAApytor\05Food\data\Bikes.zip")
# train_bikes_target = imagepath / "train" / "bikes"
# test_bikes_target = imagepath / "test" / "bikes"
# train_bikes_target.mkdir(parents=True, exist_ok=True)
# test_bikes_target.mkdir(parents=True, exist_ok=True)

# # Only copy if folders are empty
# if not any(train_bikes_target.iterdir()) and bikes_zip_path.exists():
#     temp_bikes_dir = datapath / "temp_bikes"
#     temp_bikes_dir.mkdir(exist_ok=True)
    
#     with zipfile.ZipFile(bikes_zip_path, "r") as zip_ref:
#         zip_ref.extractall(temp_bikes_dir)
    
#     train_bikes_source = temp_bikes_dir / "images.cv_93fdtqurllrwgoh8yn7ge" / "data" / "train" / "motorbike"
#     test_bikes_source = temp_bikes_dir / "images.cv_93fdtqurllrwgoh8yn7ge" / "data" / "test" / "motorbike"
    
#     if train_bikes_source.exists():
#         copy_subset(train_bikes_source, train_bikes_target, 1000)
#     if test_bikes_source.exists():
#         copy_subset(test_bikes_source, test_bikes_target, 20)
    
#     shutil.rmtree(temp_bikes_dir)
# elif not bikes_zip_path.exists():
#     print(f"Warning: Bikes zip not found at {bikes_zip_path}")

# -----------------------------
# Model
# -----------------------------
class Food(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            

        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*32*32, hidden_units),
            nn.Linear(hidden_units, output_shape)
        )
    def forward(self, x):
        return self.classifier(self.convblock2(self.convblock1(x)))

# -----------------------------
# Transforms & DataLoaders
# -----------------------------
data_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(0.5),
    TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Original + augmented datasets
ogtraindata = datasets.ImageFolder(root=traindir, transform=train_transform)
#augmented_train1 = datasets.ImageFolder(root=traindir, transform=train_transform)
#augmented_train2 = datasets.ImageFolder(root=traindir, transform=train_transform) Optional, to add distorted versions

train_og_aug = ConcatDataset([ogtraindata])

og_aug_traindataloader = DataLoader(train_og_aug, batch_size=32, shuffle=True, drop_last=True)
testdata = datasets.ImageFolder(root=testdir, transform=data_transform)
testdataloader = DataLoader(testdata, batch_size=32, shuffle=False, drop_last=False)

# -----------------------------
# Model init, loss, optimizer
# -----------------------------
food = Food(input_shape=3, hidden_units=64, output_shape=5).to(device)






class_counts = torch.tensor([986, 696, 692,1000,1000], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
lossfn = nn.CrossEntropyLoss(weight=class_weights.to(device))
optim = torch.optim.Adam(food.parameters(), lr=0.0005)

print(f"Training samples: {len(train_og_aug)}, Testing samples: {len(testdata)}")

# -----------------------------
# Training loop + store metrics
# -----------------------------
results = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}


class_counts = {classname: 0 for classname in ogtraindata.classes}

for _, label in ogtraindata.samples:
    class_name = ogtraindata.classes[label]
    class_counts[class_name] += 1

print("Number of images per class in training set:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")


epochs = 25
for epoch in range(epochs):
    trainloss, trainacc = training(
        model=food,
        dataloader=og_aug_traindataloader,
        lossfn=lossfn,
        optimizer=optim,
        device=device,
        accufn=accufn,
        print_every=3
    )
    testloss, testacc = testing(
        model=food,
        dataloader=testdataloader,
        lossfn=lossfn,
        accufn=accufn,
        device=device
    )
    
    results["train_loss"].append(trainloss)
    results["test_loss"].append(testloss)
    results["train_acc"].append(trainacc)
    results["test_acc"].append(testacc)
    
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train loss: {trainloss:.4f} | Train acc: {trainacc:.2f}% | "
          f"Test loss: {testloss:.4f} | Test acc: {testacc:.2f}%")

# -----------------------------
# Plot function
# -----------------------------
def plot_loss_curves(results):
    
    epochs = range(len(results['train_loss']))
    
    plt.figure(figsize=(15,7))
    
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, results['train_loss'], label='train_loss')
    plt.plot(epochs, results['test_loss'], label='test_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, results['train_acc'], label='train_acc')
    plt.plot(epochs, results['test_acc'], label='test_acc')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.show()








# -----------------------------
# Plot curves
# -----------------------------
#plot_loss_curves(results)

#torch.save(food.state_dict(), "food_model_weights.pth")
# Load model for inference
food = Food(input_shape=3, hidden_units=64, output_shape=5)
food.load_state_dict(torch.load("food_model_weights.pth"))
food.to(device)
food.eval()

# Prepare test samples, predictions, labels
testsamples = []
testlabels = []
predclasses = []
classnames = testdata.classes
testdataloaderflopper = DataLoader(testdata, batch_size=32, shuffle=True, drop_last=False)

for i, (X, y) in enumerate(testdataloaderflopper):
    testsamples.extend(X)
    testlabels.extend(y)
    with torch.inference_mode():
        X = X.to(device)
        logits = food(X)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        predclasses.extend(preds.cpu())
    if len(testsamples) >= 9:
        break

testsamples = testsamples[:16]
testlabels = testlabels[:16]
predclasses = predclasses[:16]

# Plot predictions
plt.figure(figsize=(16,16))
nrows, ncols = 4, 4

for i, sample in enumerate(testsamples):
    img = sample.permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(img)
    
    pred_label = classnames[predclasses[i]]
    truth_label = classnames[testlabels[i]]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    
    plt.title(title_text, fontsize=10, c="g" if pred_label==truth_label else "r")
    plt.axis(False)

plt.tight_layout()
plt.show()