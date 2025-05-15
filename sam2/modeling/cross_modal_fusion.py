import torch
from torch import nn
from typing import Tuple, List
import torch.nn.functional as F
import math
from functools import partial
from typing import Tuple, Type

import torch
from torch import nn, Tensor

from sam2.modeling.sam.mamba_block import MambaLayer
from sam2.modeling.sam.transformer import Attention, MLP
from sam2.modeling.sam2_utils import MLPDropout
from timm.models.layers import trunc_normal_


class CrossModalFusionModule(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        image_embedding_size: Tuple[int, int],
        use_feature_level: Tuple[int, ...] = (2),  # the last layer
        depth: int=3,
        bimamba: bool=False,
        use_sp_bimamba: bool=False,
        use_dwconv: bool=False,
        dropout: float=0.2,
        use_mamba_attn: bool=True,
        num_temp_pos_embed: int=3,
        pad_sequence: bool=False,
        num_ref_frames: int=3,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.image_embedding_size = image_embedding_size
        self.num_temp_pos_embed = num_temp_pos_embed
        self.pad_sequence = pad_sequence
        self.num_ref_frames = num_ref_frames

        self.transformer = TwoWayTokenTransformer(
            depth=depth,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
            attention_downsample_rate=2,
            use_mamba_before_cross_attn=use_mamba_attn,
            drop_path_rate=0.2,
            bimamba=bimamba,
            sp_bimamba=use_sp_bimamba,
            use_dwconv=use_dwconv,
            dropout=dropout,
        )
        print("CrossModalFusionModule depth=", depth)
        self.cls_token = nn.Embedding(1, transformer_dim)
        self.use_feature_level = use_feature_level
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_temp_pos_embed, transformer_dim))
        self.use_dwconv = use_dwconv

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image_embeddings: List[torch.Tensor],  # (H*W, B, C)
        image_pe: List[torch.Tensor],  # (H*W, B, C)
        text_embeddings: torch.Tensor,  # (B, N, C)
        feat_sizes: List[Tuple[int, int]],
        previous_ref_feats_list: List[List],
        previous_ref_pos_embeds_list: List[List],
        return_intermediate=False,
    ):
        output_tokens = self.cls_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            text_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, text_embeddings), dim=1)

        # read out the current level of features and organize them
        h, w = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        b, c = image_embeddings[-1].shape[1:]
        f = len(previous_ref_feats_list) + 1

        # current frame
        keys = image_embeddings[-1]
        keys_pe = image_pe[-1]
        keys_tmp_embed = self.temporal_pos_embed[:, 0:1, :].repeat(image_embeddings[-1].size(0), 1, 1)

        # previous frame
        previous_feat_list = []
        previous_pos_embed_list = []
        previous_tmp_embed_list = []

        pad_num = 0
        if self.pad_sequence:
            pad_num = self.num_ref_frames - len(previous_ref_feats_list) - 1
            f = self.num_ref_frames
            # repeat the last frame to make the number of frames equal to num_fusion_frames
            for j in range(pad_num):
                previous_feat_list.append(keys)
                previous_pos_embed_list.append(keys_pe)
                if self.num_temp_pos_embed == 3:
                    time_index = 1 if j == 0 else 2
                else:
                    time_index = pad_num + j + 1
                tmp_embed = self.temporal_pos_embed[:, time_index:time_index + 1, :].repeat(image_embeddings[-1].size(0), 1, 1)
                previous_tmp_embed_list.append(tmp_embed)

        if len(previous_ref_feats_list) != 0:
            for j in range(len(previous_ref_feats_list)):  # j is the frame index, i is the level index
                previous_feat_list.append(previous_ref_feats_list[j][-1])
                previous_pos_embed_list.append(previous_ref_pos_embeds_list[j][-1])
                # 0给current，1给previous，2给previous
                if self.num_temp_pos_embed == 3:
                    time_index = 1 if j == 0 else 2
                else:
                    time_index = pad_num + j + 1
                tmp_embed = self.temporal_pos_embed[:, time_index:time_index+1, :].repeat(image_embeddings[-1].size(0), 1, 1)
                previous_tmp_embed_list.append(tmp_embed)

        keys = torch.cat([keys, *previous_feat_list], dim=0)
        keys_pe = torch.cat([keys_pe, *previous_pos_embed_list], dim=0)
        keys_tmp_embed = torch.cat([keys_tmp_embed, *previous_tmp_embed_list], dim=0)

        keys_pe = keys_pe + keys_tmp_embed
        # hs (B, N, C), src (B, H*W, C)
        hs, src = self.transformer(keys, keys_pe, tokens, vol_sizes=(f, h, w))

        image_embeddings = src  # (B, H*W, C)
        text_embeddings = hs  # (B, N, C)
        cls_tokens = text_embeddings[:, :1, :]

        image_embeddings = image_embeddings.reshape(-1, f, h*w, c).permute(1, 2, 0, 3)

        if not return_intermediate:
            image_embeddings = image_embeddings[0]  # (f, h*w, b, c)

        return image_embeddings, cls_tokens


class TwoWayTokenTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_mamba_before_cross_attn: bool = False,
        drop_path_rate: float = 0.2,
        bimamba: bool = False,
        sp_bimamba: bool = False,
        use_dwconv: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.use_dwconv = use_dwconv
        self.layers = nn.ModuleList()

        mamba_depths = [2 for i in range(depth)]
        self.drop_path_rate = drop_path_rate
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(mamba_depths))]
        for i in range(depth):
            self.layers.append(
                TwoWayTokenAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    use_mamba_before_cross_attn=use_mamba_before_cross_attn,
                    temporal_drop_rates=dp_rates[i * 2: (i + 1) * 2],
                    bimamba=bimamba,
                    sp_bimamba=sp_bimamba,
                    use_dwconv=use_dwconv,
                    dropout=dropout,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        prompt_embedding: Tensor,
        vol_sizes: Tuple[int, int, int]=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): B x N x embedding_dim.
          image_pe (torch.Tensor): B x N x embedding_dim.
          prompt_embedding (torch.Tensor): B x N_points x embedding_dim .

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.permute(1, 0, 2)
        image_pe = image_pe.permute(1, 0, 2)

        # Prepare queries
        queries = prompt_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=prompt_embedding,
                key_pe=image_pe,
                vol_sizes=vol_sizes,
            )

        # Apply the final attention layer from the points to the image
        q = queries + prompt_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayTokenAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_mamba_before_cross_attn: bool = False,
        temporal_drop_rates: List[float] = None,
        bimamba: bool = False,
        sp_bimamba: bool = False,
        use_dwconv: bool = False,
        dropout: float = 0.2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPDropout(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation, dropout=dropout
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe
        self.use_mamba_before_cross_attn = use_mamba_before_cross_attn
        self.use_dwconv = use_dwconv
        if self.use_mamba_before_cross_attn:
            self.mamba_layers = nn.ModuleList()
            for rate in temporal_drop_rates:
                self.mamba_layers.append(
                    MambaLayer(
                        dim=embedding_dim,
                        drop_path=rate,
                        bimamba=bimamba,
                        sp_bimamba=sp_bimamba,
                        use_dwconv=use_dwconv,
                    )
                )
            print("TwoWayTokenAttentionBlock use_mamba_before_cross_attn")

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, vol_sizes: Tuple[int, int, int]=None
    ) -> Tuple[Tensor, Tensor]:  # keys (B, N, C)
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        if self.use_mamba_before_cross_attn:
            for layer in self.mamba_layers:
                keys = layer(keys, vol_sizes)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys