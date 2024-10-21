"""
Author: Dominika Ciupek
"""



import torch
import torch.nn as nn



class Block(nn.Module):
    
    """ 
    A class used to represent building units of encoder and decoder architecture.
    
    
    Attributes:
    ----------
    in_size: int
        Number of input channels.
    out_size: int
        Number of output channels.
    conv_1 and conv_2: Conv2d
        Convolutional layers.
    bn_1 and bn_2: BatchNorm2d
        Batch normalization layers.
    relu: ReLU
        Rectified linear activation function (ReLU).

    Methods:
    -------
    forward(data_in)
        Performs a sequence of operations on input data.
        Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU
    
    """    
    
    def __init__(self, in_size, out_size):
        
        """ 
        Class constructor.

        
        Parameters:
        ----------
        in_size: int
            Number of input channels.
        out_size: int
            Number of output channels.
            
        """  
        
        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_size)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
        
    def forward(self, data_in):
        
        """ 
        Performs a sequence of operations on input data.
        Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU

        
        Parameters:
        ----------
        data_in: ndarray
            Input data.
            
        Returns:
        -------
        data_out: ndarray
            Data after a sequence of operations.
        
        """  
        
        data_out = self.relu(self.bn_1(self.conv_1(data_in)))
        data_out = self.relu(self.bn_2(self.conv_2(data_out)))
        
        return data_out
    

    
class Encoder(nn.Module):
      
    """ 
    A class used to represent encoder architecture.
    
    
    Attributes:
    ----------
    in_size: int
        Number of input channels.
    out_size: int
        Number of output channels.
    conv: Block
        Building units.
    pool: MaxPool2d
        Maximum pooling layers.

    Methods:
    -------
    forward(data_in)
        Performs a sequence of operations on input data.
        Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU
        Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU => MaxPool
    
    """    
    
    def __init__(self, in_size, out_size):
        
        """ 
        Class constructor.

        
        Parameters:
        ----------
        in_size: int
            Number of input channels.
        out_size: int
            Number of output channels.
            
        """
        
        super().__init__()
        
        self.conv = Block(in_size, out_size)
        self.pool = nn.MaxPool2d((2,2))
    
    def forward(self, data_in):
        
        """ 
        Performs a sequence of operations on input data.
        Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU
        Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU => MaxPool

        
        Parameters:
        ----------
        data_in: ndarray
            Input data.
            
        Returns:
        -------
        data_out: ndarray
            Data after a sequence of operations.
        data_downsampled: ndarray
            Data after a sequence of operations and downsampling by maximum pooling.
        
        """  
         
        data_out = self.conv(data_in)
        data_downsampled = self.pool(data_out)
        
        return data_out, data_downsampled



class Decoder(nn.Module):
    
    """ 
    A class used to represent decoder architecture.
    
    
    Attributes:
    ----------
    in_size: int
        Number of input channels.
    out_size: int
        Number of output channels.
    up: ConvTranspose2d
        Transpose convolutional layers.        
    conv: Block
        Building units.

    Methods:
    -------
    forward(data_in, skip_connect)
        Performs a sequence of operations on input data.
        ConvTranspose => Concatenate => Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU
    
    """    
    
    def __init__(self, in_size, out_size):
        
        """ 
        Class constructor.

        
        Parameters:
        ----------
        in_size: int
            Number of input channels.
        out_size: int
            Number of output channels.
            
        """
        
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        self.conv = Block(out_size*2, out_size)
    
    def forward(self, data_in, skip_connect):
        
        """ 
        Performs a sequence of operations on input data.
        ConvTranspose => Concatenate => Conv => BatchNorm => ReLU => Conv => BatchNorm => ReLU

        
        Parameters:
        ----------
        data_in: ndarray
            Input data.
        skip_connect: ndarray
            Data from the encoder.
            
        Returns:
        -------
        data_out: ndarray
            Data after a sequence of operations.
        
        """
         
        data_out = self.up(data_in)
        data_out = torch.cat([data_out, skip_connect], axis=1)
        data_out = self.conv(data_out)
        
        return data_out



class ParamUNet(nn.Module):
    
    """ 
    A class used to represent UNet architecture for microstructural parameters estimation.
    
    
    Attributes:
    ----------
    in_size: int
        Number of input channels.
    features: int
        Number of model features.
    out_size: int
        Number of output channels.       
    en_1, en_2 and en_3: Encoder
        Encoders.
    de_1, de_2 and de_3: Decoder
        Decoders.
    bottleneck: Block
        Building units.
    out: Conv2d
        Convolutional layers.

    Methods:
    -------
    forward(data_in)
        Performs a sequence of operations on input data.
        Encoder => Encoder => Encoder => Bottleneck => Decoder => Decoder => Decoder => Conv
    
    """    
    
    def __init__(self, in_size, features, out_size):
        
        """ 
        Class constructor.

        
        Parameters:
        ----------
        in_size: int
            Number of input channels.
        features: int
            Number of model features.
        out_size: int
            Number of output channels.
            
        """
        
        super().__init__()
        
        self.en_1 = Encoder(in_size, features)
        self.en_2 = Encoder(features, features*2)
        self.en_3 = Encoder(features*2, features*4)
        
        self.bottleneck = Block(features*4, features*8)
        
        self.de_3 = Decoder(features*8, features*4)
        self.de_2 = Decoder(features*4, features*2)
        self.de_1 = Decoder(features*2, features)
        
        self.out = nn.Conv2d(features, out_size, kernel_size=1)
    
    def forward(self, data_in):
        
        """ 
        Performs a sequence of operations on input data.
        Encoder => Encoder => Encoder => Bottleneck => Decoder => Decoder => Decoder => Conv

        
        Parameters:
        ----------
        data_in: ndarray
            Input data.
            
        Returns:
        -------
        data_out: ndarray
            Data after a sequence of operations.
        
        """

        data_out_1, data_out = self.en_1(data_in)
        data_out_2, data_out = self.en_2(data_out)
        data_out_3, data_out = self.en_3(data_out)

        data_out = self.bottleneck(data_out)

        data_out = self.de_3(data_out, data_out_3)
        data_out = self.de_2(data_out, data_out_2)
        data_out = self.de_1(data_out, data_out_1)

        data_out = self.out(data_out)
        
        return data_out