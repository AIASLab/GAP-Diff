# Define the Generator_Prelayer, which combines the Generator Module and the preprocessing simulation Module in #2088
from .Generator import *
from .Prelayer import *
from .preprocessing_functions import *

class Generator_Prelayer(nn.Module):
    def __init__(self, args):
        super(Generator_Prelayer, self).__init__()
        self.generator = Generator(3, 3)
        self.training = args.training
        if args.training:
            self.prelayer = Prelayer(args.preprocessing_functions)
        self.noise_budget = args.noise_budget
        
    def forward(self, image):
        protective_noise = self.generator(image)
        protected_image = protective_noise * (float(self.noise_budget) / 255) + image
        protected_image = torch.clamp(protected_image, min=-1, max=+1)
        if self.training:
            preprocessed_image = self.prelayer(protected_image)
            return protected_image, preprocessed_image
        else:
            return protected_image
        
        


