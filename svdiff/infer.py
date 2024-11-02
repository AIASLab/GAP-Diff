from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import argparse
import os
from diffusers import StableDiffusionPipeline

from svdiff_pytorch import load_unet_for_svdiff, load_text_encoder_for_svdiff

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--diffusion_path",
    type=str,
    help="The diffusion path",
)


args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)

    # define prompts
    prompts = [
        "a photo of sks person",
        "a dslr portrait of sks person",
        "a photo of sks person looking at the mirror",
        "a photo of sks person in front of eiffel tower",
    ]
    # prompts = [
    #     args.prompts,
    # ]
    pretrained_model_name_or_path = args.diffusion_path
    spectral_shifts_ckpt_dir = args.model_path
    unet = load_unet_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt_dir, subfolder="unet")
    text_encoder = load_text_encoder_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt_dir, subfolder="text_encoder")
    # load pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{args.output_dir}/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)
        for i in range(5):
            images = pipe(
                [prompt] * 6,
                num_inference_steps=25
            ).images
            for idx, image in enumerate(images):
                image.save(f"{out_path}/{i}_{idx}.png")

