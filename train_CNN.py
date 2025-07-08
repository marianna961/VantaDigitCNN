import argparse, pathlib
import torch, torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
# from onnxruntime.quantization import quantize_dynamic, QuantType


class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
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
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_tf(img):
    return T.Compose([
        T.Grayscale(),
        T.Resize((img, img)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval(); loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)

def main(cfg):
    device = 'cpu'
    tf = build_tf(cfg.img)

    val_ds= datasets.ImageFolder(cfg.val_dir,  tf)

    pin = torch.cuda.is_available()
    val_ld= DataLoader(val_ds, cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=pin)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    warm = LinearLR(optimizer, 0.1, 1.0, 5)
    cos  = CosineAnnealingLR(optimizer, T_max=cfg.epochs-5, eta_min=5e-5)
    scheduler = SequentialLR(optimizer, [warm, cos], milestones=[5])

    ckpt = pathlib.Path(cfg.out) / 'best.pth'
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if cfg.resume and ckpt.exists():
        print("resuming training")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, best_acc = evaluate(model, val_ld, criterion, device)
        start_epoch = cfg.start_epoch

    else:
        best_acc = 0.0
        start_epoch = 0

    for ep in range(start_epoch, cfg.epochs):
        val_loss, val_acc = evaluate(model, val_ld, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"epoch{ep+1:02d}/{cfg.epochs}| val {val_loss:.4f} |"
              f"val acc {val_acc*100:.2f}% | lr {lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)
            print("SAVED best")

    if cfg.export_onnx:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        dummy = torch.zeros(1,1,cfg.img,cfg.img, device=device)
        onnx_p = ckpt.with_suffix('.onnx')
        torch.onnx.export(model,dummy,onnx_p,
                          input_names=['input'],output_names=['logits'],
                          dynamic_axes={'input':{0:'batch'}, 'logits':{0:'batch'}},
                          opset_version=17)
        print('saved', onnx_p)
        # int8_p = ckpt.with_name(f"{ckpt.stem}_int8.onnx")
        # quantize_dynamic(onnx_p, int8_p, weight_type=QuantType.QInt8)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default = 'dataset/train')
    p.add_argument('--val_dir', default = 'dataset/val')
    p.add_argument('--out', default='chk')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)  #6.8e-4 | 2.8e-4
    p.add_argument('--img', type=int, default=24)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--export_onnx', action='store_true')
    p.add_argument('--start_epoch', type=int, default=0)
    main(p.parse_args())
