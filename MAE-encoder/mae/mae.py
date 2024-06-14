import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .encoder import Transformer


class MAE(nn.Module):
    '''
    the implementation from https://github.com/lucidrains/vit-pytorch.
    '''
    def __init__(self,
                 *,
                 encoder,
                 decoder_dim,
                 decoder_depth=1,
                 decoder_heads=8,
                 decoder_dim_head=64,
                 device='cpu'):
        super().__init__()
        # common
        self.device =  device

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.patch_to_emb = encoder.patch_to_emb
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim,
                                   depth=decoder_depth,
                                   heads=decoder_heads,
                                   dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        batch, num_masked, masked_indices, masked_patches, encoded_tokens = self.encoder(img)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)
        
        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss