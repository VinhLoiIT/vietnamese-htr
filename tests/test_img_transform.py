import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
# from utils import ScaleImageByHeight
import glob
import random

class ScaleImageByHeight(object):
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, image):
        width, height = image.size
        factor = self.target_height / height
        new_width = int(width * factor)
        new_height = int(height * factor)
        image = image.resize((new_width, new_height), Image.NEAREST)
        return image

image_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(96),
        transforms.Grayscale(3),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

## Choose random any images
fps = glob.glob('data/VNOnDB/word/test_word/*.png')
random.seed(4)
img_paths = random.choices(fps, k=4)

origin_imgs = []
transformed_imgs = []
for i in range(4):
    origin_imgs.append(Image.open(img_paths[i]).convert('L'))
    transformed_imgs.append(image_transform(origin_imgs[i]))

max_image_row = max([image.size(1) for image in transformed_imgs])
max_image_col = max([image.size(2) for image in transformed_imgs])
padded_imgs = torch.zeros(4, 3, max_image_row, max_image_col)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
ax = axes.ravel()
for i in range(4):
    image_row = transformed_imgs[i].shape[1]
    image_col = transformed_imgs[i].shape[2]
    padded_imgs[i, :, :image_row, :image_col] = transformed_imgs[i]
    padded_img = transforms.ToPILImage()(padded_imgs[i]).convert("RGB")
    
    ax[i].imshow(origin_imgs[i], cmap='gray')
    ax[i+4].imshow(padded_img, cmap='gray')
fig.tight_layout()
plt.show()