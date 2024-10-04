# Define a GaussianBlur function for Preprocessing layer in #2088
import torch.nn as nn
from kornia.filters import GaussianBlur2d


class GaussianBlur(nn.Module):
	def __init__(self, sigma, kernel=7):
		super(GaussianBlur, self).__init__()
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

	def forward(self, input_image):
		out_image = self.gaussian_filter(input_image)
		return out_image