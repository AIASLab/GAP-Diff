# Define the whole network combines the three modules in #2088
from .Generator_Prelayer import *
from .Discriminator import Discriminator
from .Diffusion import *

class Network:
    def __init__(self, device, batch_size, lr, args, tokenizer):
        self.device = device
        self.generator_prelayer = Generator_Prelayer(args).to(device)

        self.discriminator = Discriminator(4, 3).to(device)

        self.ldm = Diffusion(args, tokenizer).to(device)
		# mark "origin" as 1, "protected" as 0
        self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

        # optimizer
        self.opt_generator_prelayer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.generator_prelayer.parameters()), lr=lr)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        # loss function
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        self.criterion_MSE = nn.MSELoss().to(device)

        # weight of GAP-Diff loss
        self.discriminator_weight = 0.001
        self.adv_weight = 1
        self.adv_weight_part1 = 0.6 


    def train(self, images: torch.Tensor, epoch):
        self.generator_prelayer.train()
        self.discriminator.train()

        with torch.enable_grad():
            # use device to compute
            images = images.to(self.device)
            protected_images, preprocessed_images= self.generator_prelayer(images)
            '''
            train discriminator
            '''
            self.opt_discriminator.zero_grad()

            # target label for image should be "origin(1)"
            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
            d_cover_loss.backward()

            #GAN : target label for protected image should be "protected"(0)
            d_label_encoded = self.discriminator(preprocessed_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
            d_encoded_loss.backward()

            self.opt_discriminator.step()

            '''
            train generator
            '''
            self.opt_generator_prelayer.zero_grad()
            #GAN : target label for protected image should be "origin"(0)
            g_label_encoded = self.discriminator(preprocessed_images)
            discriminator_loss = self.criterion_BCE(g_label_encoded, self.label_cover[:g_label_encoded.shape[0]])

            if epoch < 40:
                ldm_model_pred, ldm_target = self.ldm(preprocessed_images, self.device)
                adv_loss_part1 = self.criterion_MSE(ldm_model_pred.float(), ldm_target.float())
                total_loss = self.adv_weight * (-adv_loss_part1) + self.discriminator_weight * discriminator_loss
            else:
                protected_ldm_model_pred, protected_ldm_target = self.ldm(protected_images, self.device)
                adv_loss_part1 = self.criterion_MSE(protected_ldm_model_pred.float(), protected_ldm_target.float())  
                preprocessed_ldm_model_pred, preprocessed__ldm_target = self.ldm(preprocessed_images, self.device)
                adv_loss_part2 = self.criterion_MSE(preprocessed_ldm_model_pred.float(), preprocessed__ldm_target.float())
                total_loss = self.adv_weight_part1 * (-adv_loss_part1) + (self.adv_weight - self.adv_weight_part1) * (-adv_loss_part2) + self.discriminator_weight * discriminator_loss

            total_loss.backward()
            self.opt_generator_prelayer.step()
     
        if epoch < 40:
            result = {
                "total_loss": total_loss,
                "discriminator_loss": discriminator_loss,
                "adv_loss_part1": adv_loss_part1,
                "adv_loss_part2": torch.tensor(0.0).to(self.device)
            }
        else:
            result = {
                "total_loss": total_loss,
                "discriminator_loss": discriminator_loss,
                "adv_loss_part1": adv_loss_part1,
                "adv_loss_part2": adv_loss_part2
            }
        return result 
          
    def save_model(self, path_generator: str, path_discriminator: str):
        torch.save(self.generator_prelayer.state_dict(), path_generator)
        torch.save(self.discriminator.state_dict(), path_discriminator)

    def load_model(self, path_generator: str, path_discriminator: str):
        self.load_model_ed(path_generator)
        self.load_model_dis(path_discriminator)

