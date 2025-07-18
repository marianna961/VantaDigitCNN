import argparse
import pathlib
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.optim import AdamW, LBFGS
from torch.optim.lr_scheduler import OneCycleLR


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LitDigitClassifier(pl.LightningModule):
    def __init__(self, lr=3e-4, num_classes=10,
                 optimizer_type='adamw', max_epochs=60):
        super().__init__()
        self.save_hyperparameters()
        self.model = DigitCNN(num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    # def configure_optimizers(self):
    #     if self.hparams.optimizer_type == 'lbfgs':
    #         optimizer = LBFGS(self.parameters(), lr=self.hparams.lr)
    #         return optimizer

    #     optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
    #     warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    #     cosine = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs - 5, eta_min=5e-5)
    #     # scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
    #     scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    #     scheduler = OneCycleLR(
    #     optimizer, max_lr=3e-3, 
    #     steps_per_epoch=len(train_loader), 
    #     epochs=60)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr, weight_decay=1e-4)

        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=3e-3,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=25,
                final_div_factor=1e4
            ),
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_train_tf(img_size):
    return A.Compose([
        A.Affine(translate_percent=0.1,
                 scale=(0.9, 1.1), rotate=(-5, 5), shear=(-5, 5), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


def build_val_tf(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.ds = ImageFolder(folder)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = img.convert('L')  # gray
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img, label


def main(cfg):
    train_tf = build_train_tf(cfg.img)
    val_tf = build_val_tf(cfg.img)
    train_ds = AlbumentationsDataset(cfg.train_dir, train_tf)
    val_ds = AlbumentationsDataset(cfg.val_dir, val_tf)
    train_dl = DataLoader(train_ds,
                          batch_size=cfg.batch,
                          shuffle=True,
                          num_workers=cfg.workers)
    val_dl = DataLoader(val_ds,
                        batch_size=cfg.batch,
                        shuffle=False,
                        num_workers=cfg.workers)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.out,
        filename="best",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )
    early_stop_cb = EarlyStopping(monitor="val_acc", mode="max", patience=20)
    logger = CSVLogger(save_dir=cfg.out, name="logs")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=cfg.out,
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=10
    )

    resume_ckpt_path = Path(cfg.out) / "best.ckpt"
    if cfg.resume and resume_ckpt_path.exists():
        print("resuming training")
        model = LitDigitClassifier.load_from_checkpoint(str(resume_ckpt_path))
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_ckpt_path))
    else:
        model = LitDigitClassifier(
            lr=cfg.lr, 
            optimizer_type=cfg.optimizer_type,
            max_epochs=cfg.epochs)
        trainer.fit(model, train_dl, val_dl)

    if cfg.export_onnx:
        best_model = LitDigitClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
        best_model.eval().to("cpu")
        dummy = torch.zeros(1, 1, cfg.img, cfg.img)
        onnx_p = pathlib.Path(cfg.out) / "best.onnx"
        torch.onnx.export(
            best_model.model, dummy, onnx_p,
            input_names=['input'], output_names=['logits'],
            dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=17
        )
        print('saved', onnx_p)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default='dataset/train')
    p.add_argument('--val_dir', default='dataset/val')
    p.add_argument('--out', default='chk')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1.2e-4)
    p.add_argument('--img', type=int, default=24)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--export_onnx', action='store_true')
    p.add_argument('--optimizer_type', default='adamw')
    main(p.parse_args())
