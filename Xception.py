'''This is an implementation of the well known paper "Xception: Deep Learning with Depthwise Separable Convolutions"
You can find the original paper under the linl "https://arxiv.org/abs/1610.02357"'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D
from keras import Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D



class Convolutional_Block(keras.layers.Layer):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 convolutional_filters, 
                 stride, 
                 padding,
                 name='2D convolution layer'):
        super(Convolutional_Block, self).__init__(name = name)
        self.convolution_2D_layer=tf.keras.layers.Conv2D(input_channels, output_channels, convolutional_filters, stride, padding)
        self.BatchNormalization=tf.keras.layers.BatchNormalization(output_channels)
        self.activation_layer=tf.nn.relu()

    def call(self,input):
        """Calls the Convolutional_Block
         on the given inputs."""
        input=self.convolution_2D_layer(input)
        input=self.BatchNormalization(input)
        input=self.activation_layer(input)    
        return input

class SeparableConv2D_Block(keras.layers.Layer):
    def __init__(self, input_channels,  output_channels, convolutional_filters,stride,padding):
        super(SeparableConv2D_Block,self).__init__()
        self.convolution_2D_layer=tf.keras.layers.SeparableConv2D(input_channels,
                                                                  output_channels,
                                                                  convolutional_filters,
                                                                  padding,
                                                                  stride
                                                                  )
        self.BatchNormalization=tf.keras.layers.BatchNormalization(output_channels)
        self.activation_layer=tf.nn.relu()

    def call(self,input):
        """Calls the Convolutional_Block
         on the given inputs."""
        input=self.convolution_2D_layer(input)
        input=self.BatchNormalization(input)
        input=self.activation_layer(input)    
        return input


class ReLu_SeparableConv2D_Block(keras.layers.Layer):
    def __init__(self, input_channels,  output_channels, convolutional_filters,stride,padding):

        self.activation_layer=tf.nn.relu()
        super(SeparableConv2D_Block,self).__init__()
        self.convolution_2D_layer=tf.keras.layers.SeparableConv2D(input_channels,
                                                                  output_channels,
                                                                  convolutional_filters,
                                                                  padding,
                                                                  stride
                                                                  )
        self.BatchNormalization=tf.keras.layers.BatchNormalization(output_channels)
        

    def call(self,input):
        """Calls the Convolutional_Block
         on the given inputs."""
        input=self.convolution_2D_layer(input)
        input=self.BatchNormalization(input)
        input=self.activation_layer(input)    
        return input





def Convolutional_Block (input, num_filters,kernal_size, padding , stride=1):
    input=tf.keras.layers.Conv2D(input=input,
                                 num_filters=num_filters,
                                 kernal_size=kernal_size,
                                 padding=padding,
                                 stride=stride)
    

def SeparableConv2D_Block(input, num_filters,kernal_size, padding , stride=1):
    input=tf.keras.layers.SeparableConv2D(input=input,
                                          num_filters=num_filters,
                                          kernal_size=kernal_size,
                                          padding=padding,
                                          stride=stride
                                          )



'''Now i will start implementing the '''

class Entry_flow(keras.layers.Layer):
    def __init__(self,
                 input,
                 num_filters,
                 kernal_size, 
                 padding ,
                 stride):
       

        super(Entry_flow,self).__init__()
        #I will follow the block diagram in Figure 3 and the table 1 in page 6
        self.Block=Convolutional_Block(64, convolutional_filters=3, stride=1, padding=0),
        self.Block_x=tf.keras.layers.Conv2D(1, filters=1, stride=2, padding=0),
        self.Block_1=tf.nn.Sequential(
            Convolutional_Block(32, convolutional_filters=3, stride=1, padding=0),
            Convolutional_Block(64, convolutional_filters=3, stride=1, padding=0),
            SeparableConv2D_Block(128,3),
            ReLu_SeparableConv2D_Block(128,3),
            x_1=tf.nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            
        


        )

        def call(self, x_1,Block_x):
            Block_1 = self.Block_1(x_1)
            y=tf.keras.layers.Concatenate([x_1, Block_x], 1)
        

            return tf.keras.layers.Concatenate([x_1, Block_x], 1)
        


        self.convolution_2D_layer_1=Convolutional_Block(input,num_filters=32,kernal_size=2,stride=2)
        self.activation=tf.nn.relu()

        self.convolution_2D_layer_2=Convolutional_Block(input,num_filters=32,kernal_size=2,stride=2)
        self.activation=tf.nn.relu()
