# Define the Diffusion process for the Customize DM Module in #2088
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import PretrainedConfig
from diffusers.utils.import_utils import is_xformers_available

# define the function to load the text_encoder weight
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
    
# define the diffusion process
class Diffusion(nn.Module):
    def __init__(self, args, tokenizer):
        super(Diffusion, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
        self.text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.tokenizer = tokenizer
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.instance_prompt = args.instance_prompt
        
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
            
    def forward(self, x, device, timestep=0):
        input_ids = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(x), 1)
        
        latents = self.vae.encode(x).latent_dist.sample()  * self.vae.config.scaling_factor
        noise = torch.rand_like(latents)
        bsz = latents.shape[0]
        if timestep==0:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
        else:
            timesteps = torch.tensor([timestep] * bsz, device=latents.device, dtype=torch.long)
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.text_encoder(input_ids.to(device))[0]
        
        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        target = noise
        return model_pred, target