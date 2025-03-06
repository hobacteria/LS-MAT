from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.utils import normal_init
import numpy as np

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape
from monai.utils import ensure_tuple, ensure_tuple_rep


class AGE_Discriminator(nn.Sequential):
    """
    Patch-GAN discriminator with age embedding modification.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        channels: number of filters in the first convolutional layer (doubled for each subsequent layer)
        in_channels: number of input channels
        out_channels: number of output channels
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in the discriminator.
        kernel_size: kernel size of the convolution layers
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to batch norm.
        bias: whether to have a bias term in convolution blocks. Defaults to False.
        padding: padding to be applied to the convolutional layers
        dropout: proportion of dropout applied, defaults to 0.
        last_conv_kernel_size: kernel size of the last convolutional layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        in_channels: int,
        out_channels: int = 1,
        num_layers_d: int = 3,
        kernel_size: int = 4,
        activation: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: str | tuple = "BATCH",
        bias: bool = False,
        padding: int | Sequence[int] = 1,
        dropout: float | tuple = 0.0,
        last_conv_kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers_d = num_layers_d
        self.num_channels = channels
        self.out_channels = out_channels
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size

        # Initial convolutional layer
        self.add_module(
            "initial_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=2,
            ),
        )

        input_channels = channels
        output_channels = channels * 2

        # Intermediate layers
        for l_ in range(self.num_layers_d):
            stride = 1 if l_ == self.num_layers_d - 1 else 2
            layer = Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=stride,
            )
            self.add_module(f"layer_{l_}", layer)
            input_channels = output_channels
            output_channels *= 2

        # Final layer
        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=last_conv_kernel_size,
                in_channels=input_channels,
                out_channels=out_channels,
                bias=True,
                conv_only=True,
                padding=int((last_conv_kernel_size - 1) / 2),
                dropout=0.0,
                strides=1,
            ),
        )

        # Linear embedding for age information
        self.age_embedding = nn.Linear(1, 6**3)  # Maps scalar age to feature dimension
        self.last_embedding = nn.Linear(6**3, 6**3)  # Maps scalar age to feature dimension
        
        self.apply(normal_init)

    def forward(self, x: torch.Tensor, age: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: input tensor
            age: age tensor, scalar for each image
            target_is_real: whether the target is real or fake. If fake, 
                            the age tensor is replaced with the fake age tensor.

        Returns:
            list of intermediate features, with the last element being the output.
        """
        # Replace age with fake age if target is fake
        # Embed age and broadcast to match feature map dimensions
        
        out = [x]
        for name,submodel in self.named_children():
            if isinstance(submodel, Convolution):
                intermediate_output = submodel(out[-1])
                out.append(intermediate_output)
                #if name == f'layer_{self.num_layers_d-1}':
                if name == 'final_conv':
                    embedded_age = self.age_embedding(age).reshape(-1,self.out_channels,6,6,6)
                    
                    h = torch.mul(intermediate_output, embedded_age)
                    
                # Perform feature-wise multiplication with age embedding
        last_output = self.last_embedding(intermediate_output.flatten()).reshape(-1,self.out_channels,6,6,6)
        
        out[-1] = torch.add(last_output, h)
            
        return out[1:]


class Age_Regressor(nn.Module):
    """
    This defines a network for relating large-sized input tensors to small output tensors, ie. regressing large
    values to a prediction. An output of a single dimension can be used as value regression or multi-label
    classification prediction, an output of a single value can be used as a discriminator or critic prediction.

    The network is constructed as a sequence of layers, either :py:class:`monai.networks.blocks.Convolution` or
    :py:class:`monai.networks.blocks.ResidualUnit`, with a final fully-connected layer resizing the output from the
    blocks to the final size. Each block is defined with a stride value typically used to downsample the input using
    strided convolutions. In this way each block progressively condenses information from the input into a deep
    representation the final fully-connected layer relates to a final result.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        out_shape: tuple of integers stating the dimension of the final output tensor (minus batch dimension)
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component

    Examples::

        # infers a 2-value result (eg. a 2D cartesian coordinate) from a 64x64 image
        net = Regressor((1, 64, 64), (2,), (2, 4, 8), (2, 2, 2))

    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        modality_num=2,
        sex_num=2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: float | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels, *self.in_shape = ensure_tuple(in_shape)
        self.dimensions = len(self.in_shape)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.out_shape = ensure_tuple(out_shape)
        self.kernel_size = ensure_tuple_rep(kernel_size, self.dimensions)
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.net = nn.Sequential()
        self.modality_num = modality_num
        self.sex_num = sex_num
        echannel = self.in_channels

        padding = same_padding(kernel_size)

        self.final_size = np.asarray(self.in_shape, dtype=int)
        self.reshape = Reshape(*self.out_shape)

        # encode stage
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            layer = self._get_layer(echannel, c, s, i == len(channels) - 1)
            echannel = c  # use the output channel number as the input for the next loop
            self.net.add_module("layer_%i" % i, layer)
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)  # type: ignore
        embed_size = np.prod(self.final_size) // 2
        
        self.modality_layer = self._create_embedding_module(self.modality_num,embed_size)
        self.sex_layer = self._create_embedding_module(self.sex_num,embed_size)
        
        self.final = self._get_final_layer(np.prod((echannel, ) + self.final_size) + 2*embed_size)
    
    
    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> ResidualUnit | Convolution:
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates downsampling factor, ie. convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        layer: ResidualUnit | Convolution

        if self.num_res_units > 0:
            layer = ResidualUnit(
                subunits=self.num_res_units,
                last_conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        else:
            layer = Convolution(
                conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )

        return layer

    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model

    def _get_final_layer(self, in_shape):
        #linear = nn.Linear(int(np.prod(in_shape)), int(np.prod(self.out_shape)))
        #return nn.Sequential(nn.Flatten(), linear)
        model = nn.Sequential(nn.Linear(in_shape, int(np.prod(self.out_shape))), nn.SiLU(), nn.Linear(int(np.prod(self.out_shape)), int(np.prod(self.out_shape))))
        return model
    

    
    def get_embedding_layer(self,in_shape) -> torch.Tensor:
        linear = nn.Linear(int(np.prod(in_shape)), int(np.prod(self.out_shape)))
        return linear
    
    def forward(self, x: torch.Tensor,modality,sex) -> torch.Tensor:
        x = self.net(x)
        
        modality_embed = self.modality_layer(modality)
        sex_embed = self.sex_layer(sex)
        x = torch.cat((x.reshape(x.shape[0],-1),modality_embed,sex_embed),dim=1)
        
        x = self.final(x)
        x = self.reshape(x)
        return x



if __name__ == '__main__':
    ## for debugging
    discriminator_norm = "INSTANCE"
    discriminator = AGE_Discriminator(
        spatial_dims=3,
        num_layers_d=3,
        channels=32,
        in_channels=3,
        out_channels=1,
        norm=discriminator_norm,
    ).to('cuda')

    x = torch.rand(1, 3, 64, 64, 64).to('cuda')
    age = torch.tensor([10.]).to('cuda')
    out = discriminator(x, age)
    print(out[-1].shape)    
    print(torch.isnan(out[-1]).sum())
    
    regressor = Age_Regressor((1,64, 64, 64), (1,), (64, 128, 256), (2, 2, 2), modality_num=3,sex_num = 3).to('cuda')
    x = torch.rand(1, 1, 64, 64, 64).to('cuda')
    modality = torch.tensor([[1.,0,0]]).to('cuda')
    sex = torch.tensor([[1.,0,0]]).to('cuda')
    out = regressor(x,modality,sex)