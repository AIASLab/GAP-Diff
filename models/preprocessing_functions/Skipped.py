# Define a skipped function for Preprocessing layer in #2088
import torch
import torch.nn as nn


class Skipped(nn.Module):
	# do not do any preprocessing on the input image
	def __init__(self):
		super(Skipped, self).__init__()

	def forward(self, input_image):
		out_image = input_image
		return out_image