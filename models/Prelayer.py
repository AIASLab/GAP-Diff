# Define the Preprocessing Layer in the Preprocessing Simulation Module for #2088
import torch.nn as nn
import random
from .preprocessing_functions.Jpegmask import *
from .preprocessing_functions.Skipped import *
from .preprocessing_functions.Gaussianblur import *

class Prelayer(nn.Module):
    def __init__(self, functions):
        super(Prelayer, self).__init__()
        self.functions = nn.ModuleList(eval(functions, {'JpegMask': JpegMask, 'GaussianBlur': GaussianBlur, 'Skipped': Skipped}))  

    def forward(self, input_image):
        # randomly pick a preprocessing function
        chosen_function = random.choice(self.functions)
        # ultilize the function to preprocess the input
        preprocessed_image = chosen_function(input_image)
        return preprocessed_image
	
