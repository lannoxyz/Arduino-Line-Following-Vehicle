"""
train.py — 在电脑上训练分类模型并导出 ONNX
用法: python train.py

依赖安装:
  pip install torch torchvision onnx onnxruntime

数据集结构 (从树莓派复制过来的):
  dataset/
    Star/           *.jpg
    Octagon/        *.jpg
    Cross/          *.jpg
    Trapezium/      *.jpg
    Diamond/        *.jpg
    CuttedCircle/   *.jpg
    QuarterCircle/  *.jpg
    ArrowUp/        *.jpg
    ArrowDown/      *.jpg
    ArrowLeft/      *.jpg
    ArrowRight/     *.jpg
    RecycleSign/    *.jpg
    ButtonSign/     *.jpg
    WarningSign/    *.jpg
    QRcode/         *.jpg
    Thumb/          *.jpg
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── 配置 ──────────────────────────────────────────
DATASET_DIR   = "dataset"
OUTPUT_ONNX   = "model.onnx"
OUTPUT_LABELS = "labels.txt"
INPUT_SIZE    = 224
BATCH_SIZE    = 64     # GPU显存够用，大batch更快
EPOCHS        = 50
LR            = 1e-3
VAL_SPLIT     = 0.2
NUM_WORKERS   = 0      # Windows下设0，避免多进程开销
DEVICE        = "cuda" if torch.cuda.is_available() else \
                "mps"  if torch.backends.mps.is_available() else "cpu"
# ──────────────────────────────────────────────────

def main():
    print(f"设备: {DEVICE}  ({'GPU加速' if DEVICE == 'cuda' else 'CPU训练'})")
    print(f"数据集: {DATASET_DIR}/")
    print(f"训练轮数: {EPOCHS}  批大小: {BATCH_SIZE}  workers: {NUM_WORKERS}\n")

    # ── 数据增强 ────────────────────────────────────
    # 箭头类禁用翻转，避免方向被破坏
    ARROW_CLASSES = {"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"}

    # 通用增强（非箭头类）
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1),
            scale=(0.8, 1.2), shear=10
        ),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    # 箭头专用增强（禁止翻转，只允许小角度旋转保持方向性）
    arrow_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05),
            scale=(0.85, 1.15)
        ),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.1
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── 加载数据集 ─────────────────────────────────
    from torch.utils.data import Dataset
    from PIL import Image

    # 自定义 Dataset，按类别选择不同增强策略，预加载进内存
    class SmartDataset(Dataset):
        def __init__(self, samples, class_to_idx, is_val=False):
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
            self.is_val       = is_val
            # 预加载所有图片到内存，避免每次从硬盘读取
            print(f"  预加载 {len(samples)} 张图片到内存...")
            self.data = []
            for path, label in samples:
                img = Image.open(path).convert("RGB")
                self.data.append((img.copy(), label))
            print(f"  预加载完成")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            img, label  = self.data[i]
            cls_name    = self.idx_to_class[label]
            if self.is_val:
                return val_tf(img), label
            tf = arrow_tf if cls_name in ARROW_CLASSES else train_tf
            return tf(img), label

    ref_ds      = datasets.ImageFolder(DATASET_DIR)
    classes     = ref_ds.classes
    num_classes = len(classes)
    all_samples = ref_ds.samples
    print(f"类别 ({num_classes}): {classes}\n")

    for cls, idx in ref_ds.class_to_idx.items():
        n   = sum(1 for _, l in all_samples if l == idx)
        bar = "█" * (n // 5)
        tag = " ← 箭头(禁翻转)" if cls in ARROW_CLASSES else ""
        print(f"  {cls:<16} {n:>4} 张  {bar}{tag}")

    # 分割训练/验证
    import random
    random.shuffle(all_samples)
    val_size   = int(len(all_samples) * VAL_SPLIT)
    train_size = len(all_samples) - val_size
    train_samples = all_samples[val_size:]
    val_samples   = all_samples[:val_size]

    train_ds = SmartDataset(train_samples, ref_ds.class_to_idx, is_val=False)
    val_ds   = SmartDataset(val_samples,   ref_ds.class_to_idx, is_val=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n训练集: {train_size} 张 | 验证集: {val_size} 张\n")

    # ── 模型：MobileNetV2 迁移学习 ─────────────────
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(imgs)
                correct    += (out.argmax(1) == labels).sum().item()
                total      += len(imgs)
        return total_loss / total, correct / total

    best_val_acc = 0.0
    best_path    = "best_model.pth"

    print("=" * 60)
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValAcc':>8}  {'LR':>10}")
    print("=" * 60)

    # ── 阶段1：只训练分类头（前25轮）─────────────────
    phase1_epochs = EPOCHS // 2
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_epochs)

    for ep in range(1, phase1_epochs + 1):
        t_loss, t_acc = run_epoch(train_loader, train=True)
        _,      v_acc = run_epoch(val_loader,   train=False)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        mark   = " ★" if v_acc > best_val_acc else ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)
        print(f"{ep:>6}  {t_loss:>10.4f}  {t_acc*100:>8.1f}%  "
              f"{v_acc*100:>7.1f}%  {cur_lr:>10.2e}{mark}")

    # ── 阶段2：解冻全部层微调（后25轮）──────────────
    print("\n--- 解冻全部层，开始微调 ---\n")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - phase1_epochs)

    for ep in range(phase1_epochs + 1, EPOCHS + 1):
        t_loss, t_acc = run_epoch(train_loader, train=True)
        _,      v_acc = run_epoch(val_loader,   train=False)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        mark   = " ★" if v_acc > best_val_acc else ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)
        print(f"{ep:>6}  {t_loss:>10.4f}  {t_acc*100:>8.1f}%  "
              f"{v_acc*100:>7.1f}%  {cur_lr:>10.2e}{mark}")

    print("=" * 60)
    print(f"\n最佳验证准确率: {best_val_acc*100:.1f}%")

    # ── 导出 ONNX ─────────────────────────────────
    print(f"\n导出 ONNX → {OUTPUT_ONNX}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    torch.onnx.export(
        model, dummy, OUTPUT_ONNX,
        input_names   = ["input"],
        output_names  = ["output"],
        dynamic_axes  = {"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version = 12,
    )
    print(f"✓ 模型已导出: {OUTPUT_ONNX}")

    # ── 保存 labels.txt ───────────────────────────
    with open(OUTPUT_LABELS, "w") as f:
        for i, cls in enumerate(classes):
            f.write(f"{i} {cls}\n")
    print(f"✓ 标签已保存: {OUTPUT_LABELS}")

    # ── 验证 ONNX ─────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np
        sess     = ort.InferenceSession(OUTPUT_ONNX,
                       providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        out      = sess.run(None, {"input": dummy_np})[0]
        print(f"✓ ONNX 验证通过，输出形状: {out.shape}")
    except ImportError:
        print("  (跳过验证，可 pip install onnxruntime 后手动验证)")

    print(f"""
完成！将以下两个文件复制到树莓派同目录:
  {OUTPUT_ONNX}
  {OUTPUT_LABELS}
""")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()