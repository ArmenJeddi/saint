import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from saint.prune import saint_drop
from saint.utils import parse_prune_mode, parse_sim_threshold

class SaintBlock(Block):

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    
    def _ls1(self, x):
        return self.ls1(x) if hasattr(self, "ls1") else x
    
    def _ls2(self, x):
        return self.ls2(x) if hasattr(self, "ls2") else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, metric = self.attn(self.norm1(x))
        x = x + self._drop_path1(self._ls1(x_attn))
        
        prune_mode = self._saint_info["prune_mode"].pop(0)
        sim_threshold = self._saint_info["sim_threshold"].pop(0)
        if prune_mode:
            prune_func = saint_drop(
                    metric,
                    prune_mode,
                    sim_threshold,
                    class_token=self._saint_info["class_token"],
                    distill_token=self._saint_info["distill_token"]
            )

            x = prune_func(x)

        x = x + self._drop_path2(self._ls2(self.mlp(self.norm2(x))))
        return x


class SaintAttention(Attention):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        ) 

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k.mean(1)


def make_saint_class(transformer_class):
    class SaintVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize prune modes
        - Initialize sim threshold
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._saint_info["prune_mode"] = parse_prune_mode(len(self.blocks), self.prune_mode)
            self._saint_info["sim_threshold"] = parse_sim_threshold(len(self.blocks), self.sim_threshold)
            return super().forward(*args, **kwdargs)

    return SaintVisionTransformer


def apply_patch(model: VisionTransformer, sim_threshold, K, gamma):
    SaintVisionTransformer = make_saint_class(model.__class__)

    model.__class__ = SaintVisionTransformer
    model._saint_info = {
        "prune_mode": None,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "sim_threshold": sim_threshold,
        "K": K,
        "gamma": gamma
    }

    # TODO: check for DeiT-V3 / for now add it manually
    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._saint_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = SaintBlock
            module._saint_info = model._saint_info
        elif isinstance(module, Attention):
            module.__class__ = SaintAttention
