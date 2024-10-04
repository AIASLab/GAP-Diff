# the training process for the generator in #2088
import argparse
from pathlib import Path
import os
import logging

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoTokenizer
from tqdm import tqdm

from models.Network import *

logger = get_logger(__name__)

# prepare the input args
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train generator script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )

    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
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
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--preprocessing_functions",
        type=str,
        default= "[JpegMask(50),JpegMask(70),GaussianBlur(2),Skipped()]",
        help="Get the noise functions",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default= 4,
        help="training batch size",
    )
    parser.add_argument(
        "--resume",
        action='store_true',  
        help="if resume training",
    )
    parser.add_argument(
        "--pretrain_generator_weight",
        type=str,
        help="the path to the pretrained weight of generator ",
    )
    parser.add_argument(
        "--pretrain_discriminator_weight",
        type=str,
        help="the path to the pretrained weight of discriminator",
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

# Define a facedataset for training
class FaceDataset(Dataset):

	def __init__(self, path, H=512, W=512):
		super(FaceDataset, self).__init__()
		self.H = H
		self.W = W
		self.path = path
		self.list = os.listdir(path)
		self.transform = transforms.Compose([
			transforms.Resize((self.H, self.W)),
			transforms.RandomCrop((self.H, self.W)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	def transform_image(self, image):

		# ignore
		if image.size[0] < self.W / 2 and image.size[1] < self.H / 2:
			return None
		if image.size[0] < image.size[1] / 2 or image.size[1] < image.size[0] / 2:
			return None

		# Augment, ToTensor and Normalize
		image = self.transform(image)

		return image

	def __getitem__(self, index):

		while True:
			image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
			image = self.transform_image(image)
			if image is not None:
				return image
			# print("dataloader : skip index", index)
			index += 1

	def __len__(self):
		return len(self.list)

# Define the training function  
def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
    )

    # Dataset and DataLoaders creation:
    train_dataset = FaceDataset(args.instance_data_dir, args.resolution, args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)


    lr = 0.001
    network = Network(accelerator.device, args.train_batch_size, lr, args, tokenizer)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("G_per" + args.noise_budget, config=vars(args))

    if args.resume:
        network.load_model(args.pretrain_generator_weight, args.pretrain_discriminator_weight)
    min_total_loss = float('inf')
    path_model = args.output_dir
    path_generator = path_model + "G_per" + args.noise_budget + ".pth"
    path_discriminator = path_model + "D_per" + args.noise_budget + ".pth"

    logger.info("start training")
    for epoch in range(args.max_train_steps):
        running_result = {
                "total_loss": 0.0,
                "discriminator_loss": 0.0,
                "adv_loss_part1": 0.0,
                "adv_loss_part2": 0.0,
        }
        '''
        train
        '''
        num = 0
        for _, images, in enumerate(train_dataloader):
            image = images.to(accelerator.device)

            result = network.train(image, epoch)

            
            for key in result:
                running_result[key] += float(result[key].item())

            num += 1

            if num % 1000 == 0 or num == 1:
                logger.info("Total Loss: {}".format(running_result["total_loss"] / num))
                logger.info("Discriminator Loss: {}".format(running_result["discriminator_loss"] / num))
                logger.info("Adv Loss Part1: {}".format(running_result["adv_loss_part1"] / num))
                logger.info("Adv Loss Part2: {}".format(running_result["adv_loss_part2"] / num))
            
            
        '''
        train results
        '''
        content = "Epoch " + str(epoch) + " : " + "\n"
        for key in running_result:
            content += key + "=" + str(running_result[key] / num) + ","
        content += "\n"
        accelerator.log({"total_loss": running_result["total_loss"] / num}, step=epoch)
        accelerator.log({"discriminator_loss": running_result["discriminator_loss"] / num}, step=epoch)
        accelerator.log({"adv_loss_part1": running_result["adv_loss_part1"] / num}, step=epoch)
        accelerator.log({"adv_loss_part2": running_result["adv_loss_part2"] / num}, step=epoch)
        
        #logger.info(content)
        logger.info("***************************************************************")
        logger.info("Training epoch {} completed, loss: {}".format(epoch+1, running_result["total_loss"] / num))
        logger.info("The minimum loss before this round: {}".format(min_total_loss))
        logger.info("***************************************************************")
        if running_result["total_loss"] / num < min_total_loss:
            logger.info("Updated and saved to " + path_model + "G_per" + args.noise_budget + ".pth")
            logger.info("√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√")
            min_total_loss = running_result["total_loss"] / num
            network.save_model(path_generator,path_discriminator)

if __name__ == "__main__":
    args = parse_args()
    main(args)

