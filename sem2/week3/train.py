import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler # 混合精度加速
from torch.utils.tensorboard import SummaryWriter # monitoring
from PIL import Image
import multiprocessing

#  Config
DATASET_DIR   = "dataset"
OUTPUT_ONNX   = "model.onnx"
LOG_DIR       = "runs/experiment_1"
INPUT_SIZE    = 224
BATCH_SIZE    = 64
EPOCHS        = 40      # rounds of trainings
PATIENCE      = 5       # early stop after 5 rounds without improvements
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

ARROW_CLASSES = {"ArrowLeft", "ArrowRight", "ArrowUp"}
RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Conditional Augmentation 
def get_transforms(is_arrow=False, is_train=True):
    if not is_train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.Grayscale(3) if is_arrow else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(RGB_MEAN, RGB_STD)
        ])
    
    # 0-15 degree rotations of pictures in the dataset to increase the robustness
    base = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if is_arrow:
        base += [transforms.Grayscale(3), transforms.RandomRotation(15)]
    else:
        base += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)]
    
    base += [transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD)]
    if not is_arrow: base.append(transforms.RandomErasing(p=0.2))
    return transforms.Compose(base)

# Smart Dataset
class LannoDataset(Dataset):
    def __init__(self, samples, class_to_idx, is_val=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.is_val = is_val
        # pre-load into memory
        self.imgs = [Image.open(s[0]).convert("RGB").copy() for s in samples]

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img, label = self.imgs[i], self.samples[i][1]
        is_arrow = self.idx_to_class[label] in ARROW_CLASSES
        tf = get_transforms(is_arrow, is_train=not self.is_val)
        return tf(img), label

# trainning engine
def main():
    writer = SummaryWriter(LOG_DIR)
    ref_ds = datasets.ImageFolder(DATASET_DIR)
    num_classes = len(ref_ds.classes)
    
    # data split
    train_idx, val_idx = torch.utils.data.random_split(ref_ds.samples, [0.8, 0.2])
    train_loader = DataLoader(LannoDataset(train_idx, ref_ds.class_to_idx), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(LannoDataset(val_idx, ref_ds.class_to_idx, is_val=True), 
                            batch_size=BATCH_SIZE, num_workers=0)

    # Model and Mixed Precision Initialization
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(DEVICE)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(DEVICE)
    
    scaler = GradScaler() # container
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, epochs_no_improve = 0, 0
    
    print(f"🚀 Start Training | Device: {DEVICE} | Target: {num_classes} ")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(): 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # verification stage
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(inputs).argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_idx)
        train_acc = train_correct / len(train_idx)
        
        # monitoring records
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        print(f"Epoch {epoch:02d} | Train: {train_acc:.2%} | Val: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            epochs_no_improve = 0
            print("  🌟 Better weights found and saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  🛑 STOPPED：No improvements for {PATIENCE} rounds。")
                break
        
        scheduler.step(val_acc)

    # early stop
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    torch.onnx.export(model, dummy, OUTPUT_ONNX, input_names=['input'], output_names=['output'], opset_version=12)
    
    with open("labels.txt", "w") as f:
        [f.write(f"{i} {c}\n") for i, c in enumerate(ref_ds.classes)]
    
    print(f"\n✅ All Done! Best model with the accuracy of: {best_acc:.2%}")
    writer.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
