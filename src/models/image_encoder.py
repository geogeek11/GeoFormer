import torch
from torch import nn
from torchvision import transforms
from timm.models.swin_transformer import SwinTransformer
from timm.models.swin_transformer_v2 import SwinTransformerV2

import math
from src.models.embeddings import PolygonizerImageEmbeddings, HEATImageEmbeddings


class CustomSwinTransformer(SwinTransformer):

    def __init__(self, img_size=224, *cfg, **kwcfg):
        super(CustomSwinTransformer, self).__init__(img_size=img_size, *cfg, **kwcfg)
        self.height, self.width = img_size, img_size

    def forward_features(self, x):

        x = self.patch_embed(x)
        x = self.layers(x)

        x = self.norm(x)  # B L C

        return x


class CustomSwinTransformerV2(SwinTransformerV2):

    def __init__(
        self,
        img_size=224,
        type_of_pos_embedding="vanilla",
        softmax_output=True,
        upsampling_factor=None,
        normalize_ims=True,
        output_hidden_dims=768,
        pyramidal_ft_maps=False,
        *cfg,
        **kwargs
    ):
        super(CustomSwinTransformerV2, self).__init__(img_size=img_size, *cfg, **kwargs)

        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.normalize_ims = normalize_ims

        self.height, self.width = img_size, img_size
        self.output_hidden_dims = output_hidden_dims

        if type_of_pos_embedding not in ["prior_polygonizer"]:
            self.pos_embds_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm = nn.LayerNorm(output_hidden_dims, eps=1e-6)

        if output_hidden_dims != 768:
            self.linear_proj_to_output = nn.Linear(768, output_hidden_dims)
            self.norm = nn.LayerNorm(output_hidden_dims, eps=1e-6)

        self.num_patches = self.patch_embed.num_patches

        self.embed_dim = self.patch_embed.proj.out_channels
        self.upsampling_factor = upsampling_factor
        self.old_embedding_type = False
        self.pyramidal_ft_maps = pyramidal_ft_maps

        if self.upsampling_factor is not None:
            self.upsample_ft_maps = nn.Upsample(
                scale_factor=self.upsampling_factor, mode="bilinear"
            )

        if type_of_pos_embedding == "vanilla":
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    int(math.sqrt(self.num_patches)),
                    int(math.sqrt(self.num_patches)),
                    self.embed_dim,
                )
            )
        elif type_of_pos_embedding == "polygonizer":
            self.pos_embed = PolygonizerImageEmbeddings(
                hidden_dims=output_hidden_dims, softmax_output=softmax_output
            )  # Fixed for now, we'll make it dynamic later
        elif type_of_pos_embedding == "heat":
            self.pos_embed = HEATImageEmbeddings(
                hidden_dims=output_hidden_dims
            )  # was 768 for swin normally
        elif type_of_pos_embedding == "prior_polygonizer":
            self.pos_embed = PolygonizerImageEmbeddings(
                hidden_dims=self.embed_dim, softmax_output=softmax_output
            )
            self.old_embedding_type = True
        else:
            raise NotImplementedError

        if pyramidal_ft_maps:

            concat_channels = self.embed_dim
            concat1 = nn.Conv2d(
                96, concat_channels, kernel_size=3, padding=1, bias=False
            )
            bn1 = nn.BatchNorm2d(concat_channels)
            relu = nn.ReLU(inplace=True)

            self.conv1_concat = nn.Sequential(concat1, bn1, relu)

            concat2 = nn.Conv2d(
                192, concat_channels, kernel_size=3, padding=1, bias=False
            )
            bn2 = nn.BatchNorm2d(concat_channels)
            upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)

            self.conv2_concat = nn.Sequential(concat2, bn2, relu, upsample1)

            concat3 = nn.Conv2d(
                384, concat_channels, kernel_size=3, padding=1, bias=False
            )
            bn3 = nn.BatchNorm2d(concat_channels)
            upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)

            self.conv3_concat = nn.Sequential(concat3, bn3, relu, upsample2)

            concat4 = nn.Conv2d(
                768, concat_channels, kernel_size=3, padding=1, bias=False
            )
            bn4 = nn.BatchNorm2d(concat_channels)
            upsample3 = nn.UpsamplingBilinear2d(scale_factor=8)

            self.conv4_concat = nn.Sequential(concat4, bn4, relu, upsample3)

            final_dim = output_hidden_dims
            conv_final_1 = nn.Conv2d(
                4 * concat_channels, 128, kernel_size=3, padding=1, stride=2, bias=False
            )
            bn_final_1 = nn.BatchNorm2d(128)
            conv_final_2 = nn.Conv2d(
                128, 128, kernel_size=2, padding=3, stride=1, bias=False
            )
            bn_final_2 = nn.BatchNorm2d(128)
            conv_final_3 = nn.Conv2d(
                128, final_dim, kernel_size=2, padding=2, bias=False
            )
            bn_final_3 = nn.BatchNorm2d(final_dim)
            self.conv_final = nn.Sequential(
                conv_final_1,
                bn_final_1,
                conv_final_2,
                bn_final_2,
                conv_final_3,
                bn_final_3,
            )

            # self.pos_drop = nn.Dropout(p=self.dropout_rate)

    def forward_features(self, x, **kwargs):

        if self.normalize_ims:
            x = self.normalize(x)

        x = self.patch_embed(x)

        if self.old_embedding_type:
            x = x + self.pos_embed(x)

        if not self.pyramidal_ft_maps:
            x = self.layers(x)  # If output_layers = 3 this is Bx7x7x768
            if self.output_hidden_dims != 768:
                x = self.linear_proj_to_output(x)
            x = self.norm(x)  # B L C
        else:
            swin_l1 = self.layers[0](x)  # Bx7x7x96 -> Bx96x56s56
            swin_l2 = self.layers[1](swin_l1)  # Bx7x7x192 -> Bx192x28x28
            swin_l3 = self.layers[2](swin_l2)  # Bx7x7x384 -> Bx384x14x14
            swin_l4 = self.layers[3](swin_l3)  # Bx7x7x768 -> Bx768x7x7

            agg_ft_maps = torch.concat(
                [
                    self.conv1_concat(swin_l1.permute(0, 3, 2, 1)),
                    self.conv2_concat(swin_l2.permute(0, 3, 2, 1)),
                    self.conv3_concat(swin_l3.permute(0, 3, 2, 1)),
                    self.conv4_concat(swin_l4.permute(0, 3, 2, 1)),
                ],
                dim=1,
            )

            final_ft_maps = self.conv_final(agg_ft_maps)
            x = final_ft_maps.permute(0, 2, 3, 1)

        if self.upsampling_factor is not None:
            x = self.upsample_ft_maps(x.transpose(-1, 1)).transpose(
                1, -1
            )  # Bx7x7x768 -> Bx768x7x7 -> Bx768x14x14

        if not self.old_embedding_type:  # breaking change for now
            if isinstance(self.pos_embed, nn.Parameter):
                x = x + self.pos_embed
            else:
                # print(x.shape,self.pos_embed(x).shape)
                x = x + self.pos_embed(context=x)

        return x

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)
