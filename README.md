### Tensor basics
- `.view()` : use when you know that the tensor is contiguous and want to save time by preventing unnecessary copy. 
```python
x = torch.randn(32, 3, 224, 224)  # batch of images
x = x.view(32, -1)  # flatten each image
```
Use `.reshape()` in complex pipelines when you are not sure whether the tensor is contiguous or not.
- `.numel()` returns number of elements in the tensor, while `.shape` returns the shape of the tensor.
```python
# Finding no. of trainable params in a model -> 
sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- `.permute()`
```python
x = torch.randn(32, 3, 224, 224)   # NCHW -> PyTorch convention
x_nhwc = x.permute(0, 2, 3, 1)     # NHWC -> TensorFlow or OpenCV convention
```
- `.stack()` increases dimension by 1.
```python
img1 = torch.randn(3, 224, 224)
img2 = torch.randn(3, 224, 224)
batch = torch.stack([img1, img2], dim=0)  # shape (2, 3, 224, 224)
f1 = torch.randn(32, 64, 28, 28)  # batch, channels, H, W
f2 = torch.randn(32, 64, 28, 28)
merged = torch.cat([f1, f2], dim=1)  # shape (32, 128, 28, 28)
```
`.cat()` is used in feature fusion in multi-batch architecture, while `.stack()` is used in building a batch of images for training.
- Autograd: Calling <scaler_loss>`.backward()` stores gradients at 
	- leaf tensors (those created by user and not as the result of any operation) with `requires_grad = True`
	- Non-leaf tensor with `.retain_grad()`.

### Shape Analysis
`.shape` works on Tensors, not `Dataset` (or `Subset`) because a PyTorch `Dataset` (or a `Subset`) isn't a single block of data stored in memory—it’s more like a **map** or a **recipe book**. Do this to get their shape ->
```python
# (B, C, H, W)
# for loaders
data_iter = iter(train_loader)
images, labels = next(data_iter)
# for dataset
sample_data, sample_label = train_dataset[0]
B = labels.size(0)
B = len(train_dataset)
C, H, W = sample_data.shape # or B, C, H, W = images.shape
```
Debugging tip: Put a `print(x.shape)` inside your `forward` method during debugging.
```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        # Output: (B, 16, 32, 32) because padding=1 keeps size same for 3x3 kernel
        
        self.pool = nn.MaxPool2d(2, 2) 
        # Output: (B, 16, 16, 16)
        
        self.fc = nn.Linear(16 * 16 * 16, 10) 
        # Flattened input must match: 16 channels * 16 height * 16 width

    def forward(self, x):
	    print(x.shape) # <- Debugger
        x = self.pool(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten to (B, 4096)
        x = self.fc(x)
        print(x.shape) # <- Debugger
        return x
```
`nn.Conv2d(in_channels=I, out_channels=O, kernel_size=K, stride=S, padding=P)` expects **4D tensors**, i.e., $(B, I, N_I, N_I) \rightarrow (B, O, N_O, N_O)$ where $N_O = \left\lfloor \frac{N_I + 2P - K}{S} \right\rfloor + 1$  $\rightarrow \text{no. of params} = I * O * K^2 + O$.
`nn.Linear(in_channels=I, out_channels=O)` expect **2D tensors**, i.e., $(B, I) \rightarrow (B, O)$ $\rightarrow \text{no. of params} = I * O + O$. So, in `__init__` you need to write `nn.Linear(C * H * W, out_features)` and in `forward` you need to write `x = x.view(x.size(0), -1)`.
### Basic imports
```python
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(os.listdir())
```
### Hyperparameters
```python
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
```
### Helper function to show image
```python
def show_image(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=3)
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    plt.figure(figsize=(11, 11))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.show()
    print(labels)
```
### Helper function to train model
```python
from tqdm import tqdm
os.makedirs("./models", exist_ok=True)
def train_model(model, train_loader, val_loader, optimizer=None, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # assuming model already in cuda device
    n_total_steps = len(train_loader)
    epochs = []
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss = 0.
        val_loss = 0.
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            outputs = model(images)
            loss = criterion(outputs, labels) # averaged over the batch
            train_loss += loss.item() * batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
        train_loss /= len(train_loader.dataset)
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * batch_size
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict()
            }, f"./models/Epoch{epoch+1}.pth")
            print("Saved best model")

    print("Finished training")
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.legend()
    plt.show()
```
### Helper to evaluate model and print statistics
```python
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
def evaluate_model(model, test_loader, class_names, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print('\033[1m' + 'Printing statistics...' + '\033[0m')
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        print(f"Accuracy for class {class_name}: {class_acc[i]:.4f}")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    return test_loss, acc, f1, cm, class_acc
```
### Custom NN model
```python
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
```
### Dataset loading -> custom class inheriting `torch.utils.data.Dataset` 
The `__len__` method must return the total number of samples in the dataset. This integer value is used by the `DataLoader` to determine the length of an epoch and to generate indices for sampling.
The `__getitem__` method takes an index $idx$ and returns a single sample, usually in the form of a tuple or a dictionary containing the feature tensor and its label. This method is the primary site for lazy loading; for instance, if the dataset consists of images, `__getitem__` would load the image from disk, apply the specified transforms (such as cropping or normalization), and return the resulting tensor. This on-the-fly processing is essential for datasets that exceed the system's RAM capacity.
```python
# Extracting datasets using custom control
from PIL import Image
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_list = os.listdir(root)
        class_names = sorted(set(['_'.join(fname.split('.')[0].lower().split('_')[:-1]) for fname in self.file_list])) # your logic here
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img_name = self.file_list[index]
        img = Image.open(os.path.join(self.root, img_name))
        if self.transform:
            img = self.transform(img)
        class_name = '_'.join(img_name.split('.')[0].lower().split('_')[:-1])
        label = self.class_to_idx[class_name]
        return img, label

# finding mean and std
root = '../../../.cache/kagglehub/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/versions/1/images'
target_size = (224, 224)
calculation_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(target_size),
    torchvision.transforms.Grayscale(num_output_channels=3), # done as some images had single channel only
    torchvision.transforms.ToTensor()
])
temp_dataset = MyDataset(root = root, transform=calculation_transform)
class_to_idx = temp_dataset.class_to_idx
print(class_to_idx)
k = 1000
indices = random.sample(range(len(temp_dataset)), k=k)
processed_images = []
for i in indices:
    img, _ = temp_dataset[i]
    processed_images.append(img)
images = torch.stack(processed_images)
mean = images.mean(dim=[0, 2, 3])
std = images.std(dim=[0, 2, 3])
print(f"Mean: {mean}")
print(f"Std:  {std}")

# getting train, val and test datasets
dataset = MyDataset(root = root)
mean = [0.4480, 0.4480, 0.4480]
std = [0.2526, 0.2526, 0.2526]
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_tranform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(target_size),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
test_tranform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(target_size),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
# overriding transforms
train_dataset.dataset.transform = train_tranform
val_dataset.dataset.transform = test_tranform
test_dataset.dataset.transform = test_tranform

# getting data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
show_image(train_dataset)
```
### Getting class wise distribution
```python
from collections import Counter
labels = [train_dataset.dataset[i][1] for i in train_dataset.indices]
label_counts = Counter(labels)
plt.figure(figsize=(10, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution in Train Dataset")
plt.show()
```
### Freezing weights & setting different learning rates
model.`named_children()` gives the layer architecture of the model. To freeze, do this-> 
```python
for param in model.<layer_name>.parameters():
	param.requires_grad = False
```
And later while defining the optimizer, add the params with requires_grad as True and set their learning rate. You may use different learning rates by packing them in a dictionary like this ->
```python
optimizer = torch.optim.Adam([{'params': param_list1, 'lr': 1e-4}, {'params': param_list2, 'lr': 1e-5}, ..])
```
Complete example ->
```python
backbone_params = []
for param in model.parameters():
    param.requires_grad = False
unfreeze_layers = ['layer3', 'layer4']
for name, child in model.named_children():
    if name in unfreeze_layers:
        print(f"Unfreezing layer: {name}")
        for param in child.parameters():
            param.requires_grad = True
            backbone_params.append(param)

classifier_params = []
num_ftrs_finetune = model.fc.in_features
model.fc = nn.Linear(num_ftrs_finetune, num_classes)
for param in model.fc.parameters():
    param.requires_grad = True
    classifier_params.append(param)

model = model.to(device)
optimizer_finetune = torch.optim.Adam([
    {'params': classifier_params, 'lr': 1e-4},
    {'params': backbone_params, 'lr': 1e-5}
])
```

