# Define a JPEGMask function for Preprocessing layer in #2088
# modified from https://github.com/ando-khachatryan/HiDDeN
import torch
import torch.nn as nn
import numpy as np

class JpegMask(nn.Module):
    def __init__(self, Q, subsample=0):
        super(JpegMask, self).__init__()
        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q
        self.subsample = subsample

    #define the key JPEGMask funciton
    def round_mask(self, x):
        mask = torch.zeros(1, 3, 8, 8).to(x.device)
        mask[:, 0:1, :5, :5] = 1
        mask[:, 1:3, :3, :3] = 1
        mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
        return x * mask

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def dct(self, image):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image.shape[2] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image_dct

    def idct(self, image_dct):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image_dct.shape[2] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image

    def yuv_dct(self, image, subsample):
        image = (image.clamp(-1, 1) + 1) * 255 / 2
        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)
        image_yuv = self.rgb2yuv(image)
        image_dct = self.dct(image_yuv)
        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        image_idct = self.idct(image_quantization)
        image_ret_padded = self.yuv2rgb(image_idct)
        image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height, :image_ret_padded.shape[3] - pad_width].clone()
        return image_rgb * 2 / 255 - 1

    def forward(self, input_image):
        image_dct, pad_width, pad_height = self.yuv_dct(input_image, self.subsample)
        image_mask = self.round_mask(image_dct)
        out_image = self.idct_rgb(image_mask, pad_width, pad_height).clamp(-1, 1)
        return out_image