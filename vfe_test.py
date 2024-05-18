import os
from concurrent.futures import ProcessPoolExecutor, wait

import torch
import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import *

from VisualMixer import AttentionBlock, rank, vfe

VFE_BOX_SIZE = 14
IMAGE_BOX_W = 224
IMAGE_BOX_H = 224

GAUSSIAN_MEAN = 0.3
GAUSSIAN_STDDEV = 0.1

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

    def show_image(self, idx):
        img_name = os.path.join(self.data_dir, self.images_list[idx])
        image = Image.open(img_name)
        image.show()


def process_with_visual_mixer(dataloader, rank_module, device, vfe_box=VFE_BOX_SIZE):
    save_path = "./shuffles/mixer/" + f"{IMAGE_BOX_W}x{IMAGE_BOX_H}x{vfe_box}"
    if os.path.exists(save_path):
        [os.remove(save_path + "/" + f) for f in os.listdir(save_path)]
    else:
        os.mkdir(save_path)
    print(f"Check dir: {save_path}")

    rank_module.train()
    loop = tqdm(dataloader, total=(len(dataloader)))
    for i, (inputs_batch) in enumerate(loop):
        inputs_batch = inputs_batch.to(device)
        _, channel_w, spatial_w = rank_module(inputs_batch)
        for j in range(inputs_batch.shape[0]):
            v, _ = vfe(inputs_batch[j], vfe_box, vfe_box)
            save_image(inputs_batch[j], save_path + f"/img_{i}a_{v:.2f}.png")
            inputs_batch[j:, :, :] = rank(inputs_batch[j:, :, :], channel_w, spatial_w, vfe_box, v, [0.5, 1])
            # v, _ = vfe(inputs_batch[j], vfe_box, vfe_box)
            # save_image(inputs_batch[j], save_path + f"/img_{i}b_{v:.2f}.png")
        loop.set_description(f'Images[vm]: [{i + 1}/{len(dataloader)}]')
        print()


def process_with_gaussian_noise(dataloader, vfe_box=VFE_BOX_SIZE, mean=GAUSSIAN_MEAN, stddev=GAUSSIAN_STDDEV):
    save_path = "./shuffles/gaussian/" + f"{IMAGE_BOX_W}x{IMAGE_BOX_H}x{vfe_box}x[mean={mean},stddev={stddev}]"
    if os.path.exists(save_path):
        [os.remove(save_path + "/" + f) for f in os.listdir(save_path)]
    else:
        os.mkdir(save_path)
    print(f"Check dir: {save_path}")

    loop = tqdm(dataloader, total=(len(dataloader)))
    for i, (inputs_batch) in enumerate(loop):
        for j in range(inputs_batch.shape[0]):
            v, _ = vfe(inputs_batch[j], vfe_box, vfe_box)
            save_image(inputs_batch[j], save_path + f"/img_{i}a_{v:.2f}.png")
            noise = torch.randn(inputs_batch[j].size()) * stddev + mean
            n_image = inputs_batch[j] + noise

            v, _ = vfe(n_image, vfe_box, vfe_box)
            save_image(n_image, save_path + f"/img_{i}b_{v:.2f}.png")
        loop.set_description(f'Images[gn]: [{i + 1}/{len(dataloader)}]')
        print()


def process_with_gaussian_noise_wrap(args):
    return process_with_gaussian_noise(*args)


def process_with_visual_mixer_wrap(args):
    return process_with_visual_mixer(*args)


if __name__ == '__main__':
    device = "cpu"
    rank_module = AttentionBlock(3, 3).to(device)
    custom_transforms = transforms.Compose([
        transforms.Resize([IMAGE_BOX_H, IMAGE_BOX_W]),
        torchvision.transforms.ToTensor(),
    ])

    c_dataset = CustomDataset(data_dir="./images", transform=custom_transforms)
    c_dataloader = DataLoader(c_dataset, batch_size=1, shuffle=False)

    with ProcessPoolExecutor(max_workers=8) as pool:
        ft1 = pool.submit(process_with_visual_mixer_wrap, (c_dataloader, rank_module, device, 14))
        # ft1 = pool.submit(process_with_visual_mixer_wrap, (c_dataloader, rank_module, device, 28))
        # ft2 = pool.submit(process_with_gaussian_noise_wrap, (c_dataloader, 28, 0.0, 0.1))
        # ft3 = pool.submit(process_with_gaussian_noise_wrap, (c_dataloader, 28, 0.3, 0.1))
        # ft4 = pool.submit(process_with_gaussian_noise_wrap, (c_dataloader, 28, 0.5, 0.1))

        # ft1 = pool.submit(process_with_gaussian_noise_wrap, (c_dataloader, 28, 0.0, 0.2))
        # ft2 = pool.submit(process_with_gaussian_noise_wrap, (c_dataloader, 28, 0.0, 0.3))
        # ft3 = pool.submit(process_with_gaussian_noise_wrap, (c_dataloader, 28, 0.0, 0.5))

        wait([ft1])
