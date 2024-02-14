import re
import math
import torch
from torch import nn
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig



'''
    代码兼容性和模块替换：在某些情况下，你可能希望在不同的模型版本或实验中轻松地切换不同的层。使用 nn.Identity() 可以在不改变整体网络架构的情况下，轻松地启用或禁用某些层。例如，如果你想在实验中比较添加和不添加某个特定层的效果，可以通过将该层替换为 nn.Identity() 来快速实现这一点。

    占位符：在某些架构设计中，nn.Identity() 可以作为一个占位符存在，以保持代码的整洁和一致性。这在复杂的网络设计中尤其有用，比如在自动化的网络架构搜索或者在具有多个分支的网络中。

    保持前向函数签名的一致性：在某些复杂的模型中，可能需要保持网络层序列中每个层的输入和输出形状一致，或者需要维持特定的前向函数（forward function）签名。在这种情况下，使用 nn.Identity() 可以帮助保持这种一致性，而不改变数据。
    
    保持前向函数签名的一致性应该是主要原因
'''
class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size # ouc：2048 inc：1024
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SElayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    '''
    self.input_channels = self.adjust_channels(input_channels, width_mult) 2048
    self.kernel = kernel 3 
    self.expanded_channels = self.adjust_channels(expanded_channels, width_mult) 2048
    self.out_channels = self.adjust_channels(out_channels, width_mult) 2048
    self.use_se = use_se True
    self.use_hs = activation == "HS" True
    self.stride = stride 1
    self.dilation = dilation 1
    '''

    '''
            DPBlock(
      (mlp): Sequential(
        (0): Identity()
        (1): Linear(in_features=1024, out_features=2048, bias=True)
        (2): GELU(approximate='none')
        (3): Linear(in_features=2048, out_features=2048, bias=True)
      )
      (mb_block): Sequential(
        (0): Identity()
        (1): InvertedResidual(
          (block): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=False)
              (1): LayerNormAct2d(
                (2048,), eps=1e-05, elementwise_affine=True
                (drop): Identity()
                (act): Identity()
              )
              (2): Hardswish()
            )
            (1): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): LayerNormAct2d(
                (2048,), eps=1e-05, elementwise_affine=True
                (drop): Identity()
                (act): Identity()
              )
            )
          )
        )
        (2): InvertedResidual(
          (block): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=2048, bias=False)
              (1): LayerNormAct2d(
                (2048,), eps=1e-05, elementwise_affine=True
                (drop): Identity()
                (act): Identity()
              )
              (2): Hardswish()
            )
            (1): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): LayerNormAct2d(
                (2048,), eps=1e-05, elementwise_affine=True
                (drop): Identity()
                (act): Identity()
              )
            )
          )
        )
      )
    )
    '''
    def forward(self, x):
        b, num_tokens, c = x.shape  # x (1,576,1024)
        h = int(math.sqrt(num_tokens)) # h 24
        x = self.mlp(x) # (1,576,2048) 因为llm的feature embdedding是2048
        x = x.permute(0, 2, 1).reshape(b, -1, h, h) #(1,2048,24,24)
        x = self.mb_block(x) #(1,2048,12,12)
        x = x.flatten(2).permute(0, 2, 1) #(1,144,2048)
        return x

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

class LDPNetProjector(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)

class LDPNetV2Projector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((12, 12))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
    elif projector_type.startswith('ldpnetv2'):
        return LDPNetV2Projector(config)
    elif projector_type.startswith('ldpnet'):
        return LDPNetProjector(config)
    raise ValueError(f'Unknown projector type: {projector_type}')