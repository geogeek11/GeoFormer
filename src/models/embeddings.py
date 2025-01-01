from torch import nn
import torch
import numpy as np
import math


# class PatchEmbedding(nn.Module):
#     """
#     Performs patch embedding on an image
#     args:
#         img_size: size of the image
#         patch_size: size of the patch
#         in_chans: number of channels in the image
#         embed_dim: embedding dimension
#     returns:
#           flattened patch embeddings
#     """

#     def __init__(self, img_size, patch_size, in_chans, embed_dim):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_patches = (img_size // patch_size) ** 2
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # (B, E, P, P)
#         x = x.flatten(2)  # (B, E, N)
#         x = x.transpose(1, 2)  # (B, N, E)
#         return x


class PolygonizerImageEmbeddings(nn.Module):
    """
    From Polygonizer: https://arxiv.org/abs/2304.04048
    """

    def __init__(
        self, hidden_dims, softmax_output=False, center_context=True, device="cuda"
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.project = nn.Linear(2, self.hidden_dims)
        self.activation = nn.Softmax(dim=1) if softmax_output else nn.Identity()
        self.softmax_output = softmax_output
        self.device = device
        self.center_context = center_context

    def forward(self, context):
        image_embeddings = context.to(self.device)

        if self.center_context:
            image_embeddings = image_embeddings - 0.5

        bs, h, w, c = image_embeddings.shape

        processed_image_resolution = int(np.sqrt(h * w))

        x = torch.linspace(-1, 1, processed_image_resolution).to(self.device)

        image_coords = torch.stack(torch.meshgrid(x, x, indexing="ij"), axis=-1)

        image_coord_embeddings = self.project(image_coords).unsqueeze(0)

        image_coord_embeddings = image_coord_embeddings.repeat(bs, 1, 1, 1)

        image_embeddings = image_embeddings + image_coord_embeddings

        return self.activation(image_embeddings)


class PositionEmbeddingSine1D(nn.Module):
    """
    Taken from HEAT: https://arxiv.org/abs/2111.15143
    Position embedding for 1D sequences with distinct embeddings for even (X coordinates) and odd (Y coordinates) positions.
    Special token values are predefined, and their positions are inferred for each sequence in the batch.
    """

    def __init__(
        self,
        num_pos_feats=384,
        temperature=10000,
        normalize=False,
        scale=None,
        special_tokens=None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.special_tokens = special_tokens if special_tokens is not None else []
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        batch_size, seq_length = x.shape
        device = x.device

        # Initialize the output embedding tensor
        pos_embedding = torch.zeros(
            batch_size, seq_length, self.num_pos_feats * 2, device=device
        )

        # Temperature factor
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        for b in range(batch_size):
            # Infer the positions of special tokens in the current batch element
            special_tokens_mask = torch.isin(
                x[b], torch.tensor(self.special_tokens, device=device)
            )

            # Adjusted position indices (excluding special tokens)
            non_special_positions = torch.arange(
                seq_length, dtype=torch.float32, device=device
            )
            non_special_positions = non_special_positions[~special_tokens_mask]

            if self.normalize and len(non_special_positions) > 0:
                eps = 1e-6
                non_special_positions = (
                    non_special_positions
                    / (non_special_positions[-1] + eps)
                    * self.scale
                )

            # Compute sine and cosine for each non-special position
            pos = non_special_positions[:, None] / dim_t
            pos_sin = pos.sin()
            pos_cos = pos.cos()

            # Apply embeddings to non-special token positions
            even_positions = torch.arange(0, seq_length, 2, device=device)[
                ~special_tokens_mask[0::2]
            ]
            odd_positions = torch.arange(1, seq_length, 2, device=device)[
                ~special_tokens_mask[1::2]
            ]

            pos_embedding[b, even_positions, : self.num_pos_feats] = pos_sin[
                0 : len(even_positions), :
            ]
            pos_embedding[b, odd_positions, self.num_pos_feats :] = pos_cos[
                0 : len(odd_positions), :
            ]

        return pos_embedding


class PositionEmbeddingSine(nn.Module):
    """
    Taken from HEAT: https://arxiv.org/abs/2111.15143
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=768, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        mask = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).bool().to(x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class HEATImageEmbeddings(nn.Module):

    def __init__(self, hidden_dims=768, temperature=10000, normalize=False, scale=None):
        super().__init__()

        num_pos_feats = hidden_dims // 2

        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats, temperature, normalize, scale
        )

    def forward(self, context):

        return self.position_embedding(context).permute(0, 2, 3, 1)


class VertexEmbeddings(nn.Module):
    """
    Flexible module that allows for the use of discrete or continuous embeddings
    for use in vertex based sequence to sequence models.
    Intended to be used preliminary prior being fed to the attention layer of the Transformer/LSTM/Recurrent model.

    params:
       num_vertex_embeddings: number of discrete vertex embeddings + special tokens
       num_vertex_dimensions: number of dimensions of the vertex embeddings
       embedding_dim: dimension of the embeddings
       max_sequence_length: maximum length of the sequence
       type_of_embeddings: list of strings specifying the type of embeddings to use.
                            'vtx' for vertex embeddings
                            'pos' for positional embeddings
                            'dim' for dimension embeddings
       concat_embeddings: whether to concatenate the embeddings or add them together
       l2_norm: whether to l2 normalize the embeddings
       scale: scale factor for the positional embeddings

    """

    def __init__(
        self,
        num_vertex_embeddings=227,
        num_vertex_dimensions=2,
        embedding_dim=512,
        max_sequence_length=800,
        type_of_embeddings=["pos", "dim"],
        concat_embeddings=False,
        l2_norm=False,
        scale=1.0,
        batch_first=True,
        device="cuda",
        padtoken=2,
        max_num_objs=100,
        **kwargs
    ):
        super(VertexEmbeddings, self).__init__()

        self.num_vertex_embeddings = num_vertex_embeddings
        self.num_vertex_dimensions = num_vertex_dimensions
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.concat_embeddings = concat_embeddings
        self.scale = embedding_dim**0.5 if not l2_norm else scale
        self.l2_norm_fn = nn.Identity() if not l2_norm else nn.LayerNorm(embedding_dim)
        self.batch_first = batch_first
        self.type_of_embeddings = type_of_embeddings
        self.device = device

        if concat_embeddings:
            self.embedding_dim = embedding_dim // len(type_of_embeddings)
            assert (
                self.embedding_dim % len(type_of_embeddings) == 0
            ), "Embedding dimension must be divisible by the number of embedding types"

        if "vtx" in type_of_embeddings:
            self.vertex_embeddings = nn.Embedding(
                num_embeddings=num_vertex_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=padtoken,
            )
        else:
            self.vertex_embeddings = nn.Identity()

        if "pos" in type_of_embeddings:
            self.pos_embeddings = nn.Embedding(
                num_embeddings=max_sequence_length,
                embedding_dim=embedding_dim,
            )
        else:
            self.pos_embeddings = nn.Identity()

        if "dim" in type_of_embeddings:
            self.dimensions_embeddings = nn.Embedding(
                num_embeddings=num_vertex_dimensions, embedding_dim=embedding_dim
            )
        else:
            self.dimensions_embeddings = nn.Identity()

        if "global" in type_of_embeddings:
            self.global_inf_embeddings = nn.Embedding(
                num_embeddings=max_num_objs + 1,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )  # 0 is the padding index
        else:
            self.global_inf_embeddings = nn.Identity()

    def forward(self, vertices, global_context=None, **kwargs):

        if not self.batch_first:
            seqlen, batch_size = vertices.shape
        else:
            batch_size, seqlen = vertices.shape

        embedding_list = []

        # Handle vertex embeddings
        if "vtx" in self.type_of_embeddings:
            vertex_emb = (
                self.l2_norm_fn(self.vertex_embeddings(vertices.to(self.device)))
                * self.scale
            )
            embedding_list.append(vertex_emb)
        else:
            embedding_list.append(
                torch.zeros(batch_size, seqlen, self.embedding_dim, device=self.device)
            )

        # Handle dimension embeddings
        if "dim" in self.type_of_embeddings:
            dim_emb = (
                self.l2_norm_fn(
                    self.dimensions_embeddings(
                        torch.arange(seqlen, device=self.device)
                        % self.num_vertex_dimensions
                    )
                )
                * self.scale
            )
            embedding_list.append(dim_emb)

        # Handle positional embeddings
        if "pos" in self.type_of_embeddings:
            pos_emb = (
                self.l2_norm_fn(
                    self.pos_embeddings(torch.arange(seqlen, device=self.device))
                )
                * self.scale
            )
            embedding_list.append(pos_emb)

        # Handle global context embeddings
        if "global" in self.type_of_embeddings and global_context is not None:
            global_emb = (
                self.l2_norm_fn(self.global_inf_embeddings(global_context)) * self.scale
            )
            embedding_list.append(global_emb)

        # Combine embeddings based on the concat_embeddings flag
        if self.concat_embeddings:
            embeddings = torch.cat(embedding_list, dim=2)
        else:
            embeddings = sum(embedding_list)

        return embeddings
