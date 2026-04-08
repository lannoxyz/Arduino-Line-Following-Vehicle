import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler # 混合精度加速
from torch.utils.tensorboard import SummaryWriter # 监控看板
from PIL import Image
import multiprocessing

# ── 1. 高级配置 (Advanced Config) ──────────────────────────────────────
DATASET_DIR   = "dataset"
OUTPUT_ONNX   = "model.onnx"
LOG_DIR       = "runs/experiment_1"
INPUT_SIZE    = 224
BATCH_SIZE    = 64
EPOCHS        = 40      # 缩减总轮数
PATIENCE      = 5       # 早停耐心值：连续5轮不进步就斩断训练
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

ARROW_CLASSES = {"ArrowLeft", "ArrowRight", "ArrowUp"}
RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ── 2. 增强策略 (Conditional Augmentation) ──────────────────────────
def get_transforms(is_arrow=False, is_train=True):
    if not is_train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.Grayscale(3) if is_arrow else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(RGB_MEAN, RGB_STD)
        ])
    
    # 训练集：增加 15 度随机旋转提升鲁棒性
    base = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if is_arrow:
        base += [transforms.Grayscale(3), transforms.RandomRotation(15)]
    else:
        base += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)]
    
    base += [transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD)]
    if not is_arrow: base.append(transforms.RandomErasing(p=0.2))
    return transforms.Compose(base)

# ── 3. 核心数据集 (Smart Dataset) ───────────────────────────────────
class LannoDataset(Dataset):
    def __init__(self, samples, class_to_idx, is_val=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.is_val = is_val
        # 预加载 (按需，显存/内存平衡)
        self.imgs = [Image.open(s[0]).convert("RGB").copy() for s in samples]

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img, label = self.imgs[i], self.samples[i][1]
        is_arrow = self.idx_to_class[label] in ARROW_CLASSES
        tf = get_transforms(is_arrow, is_train=not self.is_val)
        return tf(img), label

# ── 4. 训练引擎 (The Engine) ────────────────────────────────────────
def main():
    writer = SummaryWriter(LOG_DIR)
    ref_ds = datasets.ImageFolder(DATASET_DIR)
    num_classes = len(ref_ds.classes)
    
    # 数据分割
    train_idx, val_idx = torch.utils.data.random_split(ref_ds.samples, [0.8, 0.2])
    train_loader = DataLoader(LannoDataset(train_idx, ref_ds.class_to_idx), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(LannoDataset(val_idx, ref_ds.class_to_idx, is_val=True), 
                            batch_size=BATCH_SIZE, num_workers=0)

    # 模型与混合精度初始化
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(DEVICE)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(DEVICE)
    
    scaler = GradScaler() # 混合精度缩容器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, epochs_no_improve = 0, 0
    
    print(f"🚀 开始完美训练 | 设备: {DEVICE} | 目标: {num_classes} 类")

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(): # 开启自动混合精度
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # 验证阶段
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(inputs).argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_idx)
        train_acc = train_correct / len(train_idx)
        
        # 监控记录
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        print(f"Epoch {epoch:02d} | Train: {train_acc:.2%} | Val: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # 早停与保存逻辑 (Early Stopping)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            epochs_no_improve = 0
            print("  🌟 发现更好的权重，已保存。")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  🛑 触发早停：已连续 {PATIENCE} 轮没有提升。")
                break
        
        scheduler.step(val_acc)

    # ── 5. 导出与收尾 (Export) ────────────────────────────────────────
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    torch.onnx.export(model, dummy, OUTPUT_ONNX, input_names=['input'], output_names=['output'], opset_version=12)
    
    with open("labels.txt", "w") as f:
        [f.write(f"{i} {c}\n") for i, c in enumerate(ref_ds.classes)]
    
    print(f"\n✅ 任务完成！最佳准确率: {best_acc:.2%}")
    writer.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()