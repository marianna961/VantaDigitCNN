import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

dataset_path = Path(r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset")

transform = A.Compose([
    A.Affine(translate_percent=0.05, rotate=(-10, 10), shear=(-10, 10), p=0.7),
    # A.HueSaturationValue(hue_shift_limit=10, # яркость в hvs
    #                      sat_shift_limit=10, val_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2,   # яркость в rgb
                               contrast_limit=0.2, p=0.5),
    A.CLAHE(p=0.3),
    ToTensorV2()
])


for cls in [3]:  # классы для аугментаций
    cls_dir = dataset_path / "train" / str(cls)
    imgs = list(cls_dir.glob("*.png"))
    for i, img_path in enumerate(imgs):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j in range(2):  # кол-во копий
            augmented = transform(image=img)
            aug_img_tensor = augmented["image"]
            aug_np = aug_img_tensor.permute(1, 2, 0).cpu.numpy() * 255
            aug_np = aug_np.astype("uint8")
            aug_np = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cls_dir / f"aug_{i}_{j}.png"), aug_np)
