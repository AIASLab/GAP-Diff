# the generate process for GAP-Diff in #2088
import argparse
import torch
import os
from PIL import Image
from torchvision import transforms

from models.Generator_Prelayer import *

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="generate images")
    parser.add_argument(
        "--generator_path",
        type=str,
        default=None,
        required=True,
        help="Path to the weight of generator",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        required=True,
        help="Path to source images",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=True,
        help="Path to the save file",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--noise_budget",
        type=str,
        default="16.0",
        help="the noise_budget for protective noise",
    )
    parser.add_argument(
        "--training",
        action='store_true',  
        help="if training",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda")
    generator_prelayer = Generator_Prelayer(args).to(device)
    generator_prelayer.load_state_dict(torch.load(args.generator_path))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    for filename in os.listdir(args.source_path):
        if filename.endswith('.jpg') or filename.endswith('png'):
            image_path = os.path.join(args.source_path, filename)
            image = Image.open(image_path)

            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                protected_image = generator_prelayer(image_tensor)

            protected_image = protected_image.squeeze().detach().cpu().permute(1, 2, 0).numpy()  # 去除 batch 维度，将数据移到 CPU
            protected_image = (protected_image * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            protected_image = np.clip(protected_image, 0, 255)
            protected_image = protected_image.astype(np.uint8)
            pil_image = Image.fromarray(protected_image)
            # 保存图像
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            pil_image.save(os.path.join(args.save_path, filename))
    
    print("the images are saved" )

if __name__ == "__main__":
    args = parse_args()
    main(args)



