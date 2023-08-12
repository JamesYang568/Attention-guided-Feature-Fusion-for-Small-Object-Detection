# James Yang
# jiaxiongyang@tongji.edu.cn

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, MultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


class SpatialAlignedBlock(BaseModule):
    """SAB in paper

    Args:
        parallel: using parallel decomposing conv or serial one.
        aggregate: whether add a 1*1 conv before decomposing convs
        kernel: kernel size for decomposing conv. Default 5
    """

    def __init__(self, parallel=False, aggregate=False,
                 kernel=5,
                 conv_cfg=None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: MultiConfig = dict(type='ReLU'),
                 init_cfg=None
                 ):
        super(SpatialAlignedBlock, self).__init__(init_cfg=init_cfg)
        self.parallel = parallel
        self.aggregate = aggregate
        self.aggregate = ConvModule(2,
                                    2,
                                    kernel_size=1,
                                    act_cfg=act_cfg,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg
                                    ) if aggregate else None
        self.spatial_attentive = nn.Sequential(
            ConvModule(2,
                       2,
                       kernel_size=(1, kernel),
                       padding=(0, kernel // 2),
                       act_cfg=act_cfg,
                       norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg),
            ConvModule(2,
                       2,
                       kernel_size=(kernel, 1),
                       padding=(kernel // 2, 0),
                       act_cfg=dict(type='Sigmoid'),
                       norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg)
        ) if not parallel else nn.ModuleList([  # noting this is insecure
            ConvModule(2,
                       2,
                       kernel_size=(1, kernel),
                       padding=(0, kernel // 2),
                       act_cfg=act_cfg,
                       norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg),
            ConvModule(2,
                       2,
                       kernel_size=(kernel, 1),
                       padding=(kernel // 2, 0),
                       act_cfg=dict(type='Sigmoid'),
                       norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg)
        ])

    def forward(self, shallow, deep):
        avg_h = torch.mean(deep, dim=1, keepdim=True)
        avg_l = torch.mean(shallow, dim=1, keepdim=True)
        avg_spatial = torch.concat([avg_l, avg_h], dim=1)

        if self.aggregate:
            avg_spatial = self.aggregate(avg_spatial)
        if not self.parallel:
            attention_weight = self.spatial_attentive(avg_spatial)
        else:
            attention_weight = self.spatial_attentive[1](self.spatial_attentive[0](avg_spatial))

        deep = deep * attention_weight
        shallow = shallow * (1 - attention_weight)
        # out = deep * attention_weight + shallow * (1 - attention_weight) deprecate
        return shallow, deep


class ContextualAlignedBlock(BaseModule):
    """CAB in paper
    """

    def __init__(self, shallow_channels: int, deep_channels: int,
                 ratio=4,
                 conv_cfg=None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: MultiConfig = dict(type='ReLU'),
                 init_cfg=None
                 ):
        super(ContextualAlignedBlock, self).__init__(init_cfg=init_cfg)
        self.avg_h = nn.AdaptiveAvgPool2d(1)
        self.avg_l = nn.AdaptiveAvgPool2d(1)  # easy to read :>

        feat_ch = deep_channels + shallow_channels
        self.channel_attentive = nn.Sequential(
            nn.ChannelShuffle(2),
            ConvModule(
                in_channels=feat_ch,
                out_channels=feat_ch // ratio,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=feat_ch // ratio,
                out_channels=feat_ch,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='Sigmoid')
            )
        )

    def forward(self, shallow, deep):
        avg_h = self.avg_h(deep)
        avg_l = self.avg_l(shallow)
        avg_context = torch.concat([avg_l, avg_h], dim=1)
        attention_weight = self.channel_attentive(avg_context)
        deep = attention_weight * deep
        shallow = (1 - attention_weight) * shallow
        # out = deep * attention_weight + shallow * (1 - attention_weight)
        return shallow, deep


@MODELS.register_module()
class AttentionAlignedFeatureFusion(BaseModule):
    def __init__(self, shallow_channels, deep_channels,
                 parallel=False, aggregate=True, kernel=5,
                 ratio=4,
                 conv_cfg=None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: MultiConfig = dict(type='ReLU'),
                 init_cfg=None):
        super(AttentionAlignedFeatureFusion, self).__init__(init_cfg=None)
        self.SABlock = SpatialAlignedBlock(parallel=parallel, aggregate=aggregate, kernel=kernel,
                                           conv_cfg=conv_cfg,
                                           norm_cfg=norm_cfg,
                                           act_cfg=act_cfg,
                                           init_cfg=init_cfg)
        self.CABlock = ContextualAlignedBlock(shallow_channels, deep_channels,
                                              ratio=ratio,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg,
                                              act_cfg=act_cfg,
                                              init_cfg=init_cfg)

    def forward(self, shallow, deep):
        s_aligned_shallow, s_aligned_deep = self.SABlock(shallow, deep)
        c_aligned_shallow, c_aligned_deep = self.CABlock(shallow, deep)
        shallow = s_aligned_shallow + c_aligned_shallow
        deep = s_aligned_deep + c_aligned_deep
        out = torch.concat([deep, shallow], dim=1)
        return out


@MODELS.register_module()
class FeatureSupplementModule(BaseModule):
    """Attention-Guided Shallow Feature Supplement(SFSM)

    Args:
        ratio: reduction of channels
        dilations: using dilate conv, [1] if not.
        share: using shared MLP. Default False.
        fix: whether using concat output for bottom-up path
    """

    def __init__(self, shallow_channels: int, deep_channels: int,
                 ratio=4,
                 dilations=[2, 4],
                 share=False,
                 fix=True,
                 conv_cfg=None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: MultiConfig = dict(type='ReLU'),
                 init_cfg=None):

        super(FeatureSupplementModule, self).__init__(init_cfg=init_cfg)
        self.fix = fix
        plane = shallow_channels // ratio
        self.down_ = ConvModule(shallow_channels,
                                shallow_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                act_cfg=act_cfg,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg
                                )
        self.spatial_attention = nn.Sequential(
            ConvModule(shallow_channels,  # you can use conv to shrink feature map which is same as self.down_
                       plane,
                       kernel_size=1,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg,
                       conv_cfg=conv_cfg
                       ),
            *[ConvModule(plane, plane,
                         kernel_size=3,
                         dilation=dilation,
                         padding=dilation,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg
                         ) for dilation in dilations],
            ConvModule(plane,
                       1,
                       kernel_size=1,
                       norm_cfg=None,  # no need for normalization
                       act_cfg=dict(type='Sigmoid')
                       )
        )
        # self.channel_attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     ConvModule(deep_channels,
        #                deep_channels // ratio,
        #                kernel_size=1,
        #                act_cfg=act_cfg,
        #                norm_cfg=norm_cfg,
        #                conv_cfg=conv_cfg
        #                ),
        #     ConvModule(deep_channels // ratio,
        #                deep_channels,
        #                kernel_size=1,
        #                act_cfg=dict(type='Sigmoid'),
        #                conv_cfg=conv_cfg,
        #                norm_cfg=norm_cfg
        #                )
        # )

        self.max_p = nn.AdaptiveMaxPool2d(1)
        self.avg_p = nn.AdaptiveAvgPool2d(1)
        self.share = share
        self.channel_attention = nn.ModuleList(
            nn.Sequential(
                ConvModule(deep_channels,
                           deep_channels // ratio,
                           kernel_size=1,
                           act_cfg=act_cfg,
                           norm_cfg=norm_cfg,
                           conv_cfg=conv_cfg
                           ),
                ConvModule(deep_channels // ratio,
                           deep_channels,
                           kernel_size=1,
                           act_cfg=None,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg
                           )
            )
        )
        if not share:
            self.channel_attention.append(
                nn.Sequential(
                    ConvModule(deep_channels,
                               deep_channels // ratio,
                               kernel_size=1,
                               act_cfg=act_cfg,
                               norm_cfg=norm_cfg,
                               conv_cfg=conv_cfg
                               ),
                    ConvModule(deep_channels // ratio,
                               deep_channels,
                               kernel_size=1,
                               act_cfg=None,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg
                               )
                )
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, shallow, deep):
        """
        Args:
            shallow: with shallow structural information for spatial enhancement
            deep: with deep contextual information for classification

        Returns:
            fixing output P3 same as FPN using concat op.
            or
            unfix outputs like P3 and Fb2t in paper
        """
        spatial_att = self.spatial_attention(shallow)

        channel_max = self.max_p(deep)
        channel_avg = self.avg_p(deep)
        if self.share:
            channel_avg = self.channel_attention[0](channel_avg)
            channel_max = self.channel_attention[0](channel_max)
        else:
            channel_avg = self.channel_attention[0](channel_avg)
            channel_max = self.channel_attention[1](channel_max)
        channel_att = self.sigmoid(channel_max + channel_avg)

        shallow = shallow * channel_att
        shallow = self.down_(shallow)

        deep_ = deep * spatial_att

        out = torch.concat([deep_, shallow], dim=1)
        deep = deep + deep_
        if self.fix:
            return out
        else:
            return deep, out
