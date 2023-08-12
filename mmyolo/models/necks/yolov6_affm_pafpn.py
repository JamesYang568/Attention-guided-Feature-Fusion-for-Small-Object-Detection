# James Yang
# jiaxiongyang@tongji.edu.cn

from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .attention_guided_yolo_neck import AttentionGuidedYOLONeck
from ..layers import RepStageBlock
from ..utils import make_round


@MODELS.register_module()
class YOLOv6RepPAFPN_AFFM(AttentionGuidedYOLONeck):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 12,
                 affm_cfg: ConfigType = None,
                 freeze_all: bool = False,
                 finer_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 init_cfg: OptMultiConfig = None
                 ):
        self.num_csp_blocks = num_csp_blocks
        self.block_cfg = block_cfg
        self.affm_cfg = affm_cfg
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            finer_cfg=finer_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = ConvModule(
                in_channels=int(self.in_channels[idx] * self.widen_factor),
                out_channels=int(self.out_channels[idx - 1] *
                                 self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer.
        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The upsample layer.
        """
        return nn.ConvTranspose2d(
            in_channels=int(self.out_channels[idx - 1] * self.widen_factor),
            out_channels=int(self.out_channels[idx - 1] * self.widen_factor),
            kernel_size=2,
            stride=2,
            bias=True)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = self.block_cfg.copy()

        layer0 = RepStageBlock(
            in_channels=int(
                (self.out_channels[idx - 1] + self.in_channels[idx - 1]) *
                self.widen_factor),
            out_channels=int(self.out_channels[idx - 1] * self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            block_cfg=block_cfg)

        if idx == 1:
            return layer0
        elif idx == 2:
            layer1 = ConvModule(
                in_channels=int(self.out_channels[idx - 1] *
                                self.widen_factor),
                out_channels=int(self.out_channels[idx - 2] *
                                 self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            return nn.Sequential(layer0, layer1)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            in_channels=int(self.out_channels[idx] * self.widen_factor),
            out_channels=int(self.out_channels[idx] * self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=3 // 2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = self.block_cfg.copy()

        return RepStageBlock(
            in_channels=int(self.out_channels[idx] * 2 * self.widen_factor),
            out_channels=int(self.out_channels[idx + 1] * self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            block_cfg=block_cfg)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()

    def build_affm(self, s_channels: int, d_channels: int) -> nn.Module:
        """build affm layer."""
        self.affm_cfg['shallow_channels'] = s_channels
        self.affm_cfg['deep_channels'] = d_channels
        return MODELS.build(self.affm_cfg)

    def init_weights(self):
        if self.init_cfg is None:
            """Initialize the parameters."""
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()
