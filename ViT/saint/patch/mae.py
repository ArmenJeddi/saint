import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from saint.utils import parse_prune_mode, parse_sim_threshold
from .timm import SaintBlock, SaintAttention

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

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # From the MAE implementation
            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)
                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            return outcome

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

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = SaintBlock
            module._saint_info = model._saint_info
        elif isinstance(module, Attention):
            module.__class__ = SaintAttention
