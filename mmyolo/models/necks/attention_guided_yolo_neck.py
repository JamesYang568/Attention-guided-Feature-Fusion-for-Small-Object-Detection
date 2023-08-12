# James Yang
# jiaxiongyang@tongji.edu.cn

from abc import ABCMeta, abstractmethod
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType, MultiConfig
from mmengine.model import BaseModule

from mmyolo.models.layers import PSABlock
from mmyolo.registry import MODELS


@MODELS.register_module()
class AttentionGuidedYOLONeck(BaseModule, metaclass=ABCMeta):
    """This class is the higher wrapper for BaseYOLONeck and you can see it as BaseYOLONeck either

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        finer_cfg (ConfigType): the finer module to use, which is usually a Attention Mechanism.
            Defaults to None, which means same as BaseYOLONeck.
        ...
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[int, List[int]],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 upsample_feats_cat_first: bool = True,
                 freeze_all: bool = False,
                 finer_cfg: ConfigType = None,  # 注意力机制的配置
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs):

        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        self.affm_layers_top_down = nn.ModuleList()
        self.affm_layers_down_top = nn.ModuleList()

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        # build affm blocks
        for idx in range(1, len(in_channels)):
            self.affm_layers_top_down.append(self.build_affm(in_channels[idx], out_channels[idx]))
            self.affm_layers_down_top.append(self.build_affm(out_channels[idx - 1], out_channels[idx - 1]))

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

        self.finer_cfg = finer_cfg
        if self.finer_cfg is not None:
            self.finer_module = MODELS.build(finer_cfg)
        else:
            self.finer_module = None

    @abstractmethod
    def build_reduce_layer(self, idx: int):
        """build reduce layer."""
        pass

    @abstractmethod
    def build_upsample_layer(self, idx: int):
        """build upsample layer."""
        pass

    @abstractmethod
    def build_top_down_layer(self, idx: int):
        """build top down layer."""
        pass

    @abstractmethod
    def build_downsample_layer(self, idx: int):
        """build downsample layer."""
        pass

    @abstractmethod
    def build_bottom_up_layer(self, idx: int):
        """build bottom up layer."""
        pass

    @abstractmethod
    def build_out_layer(self, idx: int):
        """build out layer."""
        pass

    @abstractmethod
    def build_affm(self, s_channels: int, d_channels: int):  #
        """build attention-guided feature fusion layer."""
        pass

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        if self.finer_cfg is not None:
            assert len(inputs) == len(self.in_channels) + 1  # ask for C2
        else:
            return super(AttentionGuidedYOLONeck, self).forward(inputs)  # father class

        C2 = inputs[0]
        inputs = inputs[1:]

        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):  # reduce dimension
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]  # keep in-channel_size
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                self.affm_layers_top_down[len(self.in_channels) - 1 - idx](feat_low, upsample_feat))  # replace concat
            inner_outs.insert(0, inner_out)

        # for shallow feature supplement
        if self.finer_cfg['fix']:
            inner_outs[0] = self.finer_module(C2, inner_outs[0])  # concat
        else:
            inner_outs[0], last_out = self.finer_module(C2, inner_outs[0])  # deep + deep_, concat

            # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                self.affm_layers_down_top[idx](downsample_feat, feat_high))  # replace concat
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        if not self.finer_cfg['fix']:
            results[0] = last_out

        return tuple(results)
