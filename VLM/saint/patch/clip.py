import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from saint.prune import (bipartite_soft_matching, random_drop_last, cluster_and_select_tokens_itself,
                         kmeans_cluster_with_cls, iterative_token_drop, iterative_token_drop_merge,
                         dbscan_with_cls_cosine, cluster_with_cls_keys, iterative_token_drop_with_l2,
                         dbscan_with_cls_batch, iterative_drop_full_graph)
from saint.utils import parse_prune_mode, parse_sim_threshold
import math
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPVisionTransformer
from typing import List, Tuple, Union, Optional
from torch import nn

class CLIPSaintBlock(CLIPEncoderLayer):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # print("FORWARDING DEADPOOL")
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, metric = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        
        prune_mode = self._saint_info["prune_mode"].pop(0)
        sim_threshold = self._saint_info["sim_threshold"].pop(0)
        if prune_mode:
            if prune_mode == 'iterative_drop_full_graph':
                keep_num = 376
                r = 16
                hidden_states = iterative_drop_full_graph(hidden_states, metric, keep_num=keep_num, r=r)
            else:
                prune_func = bipartite_soft_matching(
                    metric,
                    prune_mode,
                    sim_threshold,
                    self._saint_info["class_token"],
                    self._saint_info["distill_token"]
                )
                hidden_states = prune_func(hidden_states)
        

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    

### TODO - return k only when needed
class CLIPSaintAttention(CLIPAttention):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        original_queries = query_states.clone().detach()
        original_qeury_states = self._shape(query_states, -1, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        original_value_states = self._shape(query_states, -1, bsz)

        original_shape = key_states.shape
        original_keys = key_states.clone().detach()
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        original_values = value_states.clone().detach()
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        # key_states = key_states.view(original_shape)
        # print(f"key_states.shape {key_states.shape}")
        # print(f"original_shape.shape {original_shape}")
        # print(f"original_shape.mean(1).shape {original_keys.mean(1).shape}")
        # print(f"key_states.shape {key_states.shape}")
        # print(f"key_states.mean(0).shape {key_states.mean(0).unsqueeze(0).shape}")
        # print(f"original_shape.mean(1).shape {original_keys.mean(1).shape}")
        # print(f"----------------- original_queries {original_qeury_states.shape}")
        # return attn_output, attn_weights_reshaped, original_qeury_states.mean(1)
        # return attn_output, attn_weights_reshaped, original_value_states.mean(1)
        # return attn_output, attn_weights_reshaped, original_queries.mean(1)
        # return attn_output, attn_weights_reshaped, original_values.mean(1)
        return attn_output, attn_weights_reshaped, original_keys.mean(1)
        # original_keys = original_keys.permute(0, 2, 1, 3) 
        # B, T, H, D = original_keys.shape
        # original_keys = original_keys.reshape(B, T, H * D) 
        # return attn_output, attn_weights_reshaped, original_keys


def make_saint_class(transformer_class):
    class SaintVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize prune modes
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._saint_info["prune_mode"] = parse_prune_mode(len(self.encoder.layers), self.prune_mode)
            self._saint_info["sim_threshold"] = parse_sim_threshold(len(self.encoder.layers), self.sim_threshold)

            return super().forward(*args, **kwdargs)

    return SaintVisionTransformer


def apply_patch(model: CLIPVisionTransformer, threshold=0.9):

    SaintVisionTransformer = make_saint_class(model.__class__)

    model.__class__ = SaintVisionTransformer
    model.prune_mode = None
    model.sim_threshold = 1.0
    model.class_token = None
    model._saint_info = {
        "prune_mode": model.prune_mode,
        "class_token": model.embeddings.class_embedding is not None,
        "distill_token": False,
        "sim_threshold": threshold
    }

    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module.__class__ = CLIPSaintBlock
            module._saint_info = model._saint_info
        elif isinstance(module, CLIPAttention):
            module.__class__ = CLIPSaintAttention
    