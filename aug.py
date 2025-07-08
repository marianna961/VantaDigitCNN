from torchvision.utils import save_image
import os
from PIL import Image
import glob
import torchvision.transforms as transforms
dataset_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset"
transform = transforms.Compose([transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)])
for cls in [8, 9]: # классы для аугментаций
    imgs = glob.glob(os.path.join(dataset_path, "val", str(cls), "*.png"))
    # print(len(imgs))
    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        for j in range(1):  # кол-во копий
            aug_img = transform(img_tensor)
            save_image(aug_img, os.path.join(dataset_path, "val", str(cls), f"aug_{i}_{j}.png"))