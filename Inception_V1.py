''' This is an implementation of a research paper called "Going deeper with convolutions" which discussed
    the idea of Inception V1 Architecture''' 
''' The paper could be found here "https://arxiv.org/abs/1409.4842" '''

'''The architecture of GoogleNet is consist mainly from four classes which are: 
   1-Convolutional Blocks
   2- Auxiliary Blocks
   3-Inception Blocks
   4- The main GoogleNet '''

''' The network is 22 layers deep when counting only layers with parameters (or 27 layers if we also count pooling).
    The overall number of layers (independent building blocks) used for the construction of the network is about 
    100'''
'''I will implement GoogleNet by using Tensorflow subcalssing'''

'''I will start by importing the main packages that i need'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt


#tf.test.is_gpu_available(
 #   cuda_only=False, min_cuda_compute_capability=None
#)


'''Now, i will implement the first class which is the Convolutional Block. As written in the paper, each Convolutional_Block consist of
1- Convolutional.
2- BatchNormalization layer.
3- Activation (ReLu) layer.
   This class takes as arguments:
1-input_channels 
2- out_channels 
3- convolutional_filters. 
4-stride.
5-padding. '''

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


# Let us implement the auxiliary classifiers as mentioned in the paper        
'''By adding auxiliary classifiers connected to these intermediate layers, we would expect to encourage 
   discrimination in the lower stages in the classifier, increase the gradient signal that gets propagated
   back, and provide additional regularization. These classifiers take the form of smaller convolutional 
   networks put on top of the output of the Inception (4a) and (4d) modules. During training, their loss
   gets added to the total loss of the network with a discount weight (the losses of the auxiliary classifiers
   were weighted by 0.3). At inference time, these auxiliary networks are discarded. The exact structure of
   the extra network on the side, including the auxiliary classifier, is as follows:
   1- An average pooling layer with 5*5 filter size and stride 3, resulting in an 4*4*512 output
      for the (4a), and 4*4*528 for the (4d) stage.
   2- A 1*1 convolution with 128 filters for dimension reduction and rectified linear activation.
   3- A fully connected layer with 1024 units and rectified linear activation.
   4- A dropout layer with 70% ratio of dropped outputs.
   5- A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the
      main classifier, but removed at inference time).  '''        

class Auxiliary_Block(keras.layers.Layer):
    def __init__(self, input_channels, number_classes):
        super(self,Auxiliary_Block).__init__()

        self.Adaptive_Avg_Pool2d = tf.nn.AdaptiveAvgPool2d((4, 4))
        self.convolution_2D_layer = tf.nn.Conv2d(input_channels, 128, convolutional_filters=1, stride=1, padding=0)
        self.activation_layer = tf.nn.ReLU()

        self.fully_conn_1 = tf.nn.Linear(2048, 1024)
        self.dropout = tf.nn.Dropout(0.7)
        self.fully_conn_2 = tf.nn.Linear(1024, number_classes)

    def call(self, input):
        output = self.Adaptive_Avg_Pool2d(input)

        output = self.convolution_2D_layer(output)
        output = self.activation_layer(output)
        print('out shape is  ', output.shape)
        # out shape is  torch.Size([2, 128, 4, 4])

        output = tf.keras.layers.Flatten(output, 1)

        output = self.fully_conn_1(output)
        output = self.activation_layer(output)
        output = self.dropout(output)

        output = self.fully_conn_2(output)

        return output



'''Now i will implement the Inception Block.
From the paper: 

As these “Inception modules” are stacked on top of each other, their output correlation statistics
are bound to vary: as features of higher abstraction are captured by higher layers, their spatial
concentration is expected to decrease suggesting that the ratio of 3*3 and 5*5 convolutions should
increase as we move to higher layers.
One big problem with the above modules, at least in this naive form, is that even a modest number of
5*5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number
of filters. This problem becomes even more pronounced once pooling units are added to the mix:
their number of output filters equals to the number of filters in the previous stage. The merging of
the output of the pooling layer with the outputs of convolutional layers would lead to an inevitable
increase in the number of outputs from stage to stage. Even while this architecture might cover the
optimal sparse structure, it would do it very inefficiently, leading to a computational blow up within
a few stages.
This leads to the second idea of the proposed architecture: judiciously applying dimension reductions
and projections wherever the computational requirements would increase too much otherwise.
This is based on the success of embeddings: even low dimensional embeddings might contain a lot
of information about a relatively large image patch. However, embeddings represent information in
a dense, compressed form and compressed information is harder to model. We would like to keep
our representation sparse at most places (as required by the conditions of [2]) and compress the
signals only whenever they have to be aggregated en masse. That is, 1*1 convolutions are used to
compute reductions before the expensive 3*3 and 5*5 convolutions. Besides being used as reductions,
they also include the use of rectified linear activation which makes them dual-purpose. The
final result is depicted in Figure 2(b).
In general, an Inception network is a network consisting of modules of the above type stacked upon
each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid. For
technical reasons (memory efficiency during training), it seemed beneficial to start using Inception
modules only at higher layers while keeping the lower layers in traditional convolutional fashion.
This is not strictly necessary, simply reflecting some infrastructural inefficiencies in our current
implementation. '''   

class Inception_Block(keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 num1x1,
                 Reduction_layer_filters_3_by_3,
                 num3x3,
                 Reduction_layer_filters_5_by_5,
                 num5x5,
                 pooling_projectionn,):

        super(Inception_Block,self).__init__()

        #I will follow the block diagram in Figure 3 and the table 1 in page 6
        self.Block_1=tf.nn.Sequential(
            Convolutional_Block(input_channels, num1x1, convolutional_filters=1, stride=1, padding=0)
        )
        self.Block_2=tf.nn.Sequential(
            Convolutional_Block(input_channels, Reduction_layer_filters_3_by_3, convolutional_filters=1, stride=1, padding=0),
            Convolutional_Block(Reduction_layer_filters_3_by_3, num3x3, convolutional_filters=3, stride=1, padding=1)
        )
        self.Block_3=tf.nn.Sequential(
            Convolutional_Block(input_channels, Reduction_layer_filters_5_by_5, convolutional_filters=1, stride=1, padding=0),
            Convolutional_Block(Reduction_layer_filters_5_by_5, num5x5, convolutional_filters=5, stride=1, padding=2),
        )
        self.Block_4=tf.nn.Sequential(
            tf.nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            Convolutional_Block(input_channels,pooling_projectionn, convolutional_filters=1, stride=1, padding=0),
        )
        def call(self, input):
        # We need to invoke the Blocks. All of them taking the saem input at the beggining and then 
        # we have to concatenate them all together according to the block diagram in the paper

          Block_1 = self.Block_1(input)
          Block_2 = self.Block_2(input)
          Block_3 = self.Block_3(input)
          Block_4 = self.Block_4(input)
          #concatenate them all together as in the paper
          return tf.keras.layers.Concatenate([Block_1, Block_2, Block_3, Block_4], 1)



# Building the Inception_V1 from the block diagram as in the paper

class Inception_V1(keras.layers.Layer):
    def __init__(self, number_classes=10):
        super(Inception_V1, self).__init__()

        self.First_Conv_Block = Convolutional_Block(3, 64, convolutional_filters=7, stride=2, padding=3)
        self.Max_pooling_operation_1st = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.Second_Conv_Block = Convolutional_Block(64, 64, convolutional_filters=1, stride=1, padding=0)
        self.Third_Conv_Block = Convolutional_Block(64, 192, convolutional_filters=3, stride=1, padding=1)
        self.Max_pooling_operation_2nd = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        # We have 9 inception block in the paper 
        self.Inception_Block_1 = Inception_Block(
            input_channels=192,
            num1x1=64,
            Reduction_layer_filters_3_by_3=96,
            num3x3=128,
            Reduction_layer_filters_5_by_5=16,
            num5x5=32,
            pooling_projectionn=32,
        )
        self.Inception_Block_2 = Inception_Block(
            input_channels=256,
            num1x1=128,
            Reduction_layer_filters_3_by_3=128,
            num3x3=192,
            Reduction_layer_filters_5_by_5=32,
            num5x5=96,
            pooling_projectionn=64,
        )
        self.Max_pooling_operation_3rd = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.Inception_Block_3 = Inception_Block(
            input_channels=480,
            num1x1=192,
            Reduction_layer_filters_3_by_3=96,
            num3x3=208,
            Reduction_layer_filters_5_by_5=16,
            num5x5=48,
            pooling_projectionn=64,
        )
        self.Inception_Block_4 = Inception_Block(
            input_channels=512,
            num1x1=160,
            Reduction_layer_filters_3_by_3=112,
            num3x3=224,
            Reduction_layer_filters_5_by_5=24,
            num5x5=64,
            pooling_projectionn=64,
        )
        self.Inception_Block_5 = Inception_Block(
            input_channels=512,
            num1x1=128,
            Reduction_layer_filters_3_by_3=128,
            num3x3=256,
            Reduction_layer_filters_5_by_5=24,
            num5x5=64,
            pooling_projectionn=64,
        )
        self.Inception_Block_6 = Inception_Block(
            input_channels=512,
            num1x1=112,
            Reduction_layer_filters_3_by_3=144,
            num3x3=288,
            Reduction_layer_filters_5_by_5=32,
            num5x5=64,
            pooling_projectionn=64,
        )
        self.Inception_Block_7 = Inception_Block(
            input_channels=528,
            num1x1=256,
            Reduction_layer_filters_3_by_3=160,
            num3x3=320,
            Reduction_layer_filters_5_by_5=32,
            num5x5=128,
            pooling_projectionn=128,
        )
        self.Max_pooling_operation_4th = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.Inception_Block_8 = Inception_Block(
            input_channels=832,
            num1x1=256,
            Reduction_layer_filters_3_by_3=160,
            num3x3=320,
            Reduction_layer_filters_5_by_5=32,
            num5x5=128,
            pooling_projectionn=128,
        )
        self.Inception_Block_9 = Inception_Block(
            input_channels=832,
            num1x1=384,
            Reduction_layer_filters_3_by_3=192,
            num3x3=384,
            Reduction_layer_filters_5_by_5=48,
            num5x5=128,
            pooling_projectionn=128,
        )
        self.Max_pooling_operation_5th = tf.nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = tf.nn.Dropout(0.4)
        self.fc = tf.nn.Linear(1024, number_classes)

        self.Auxiliary_Block_1 = Auxiliary_Block(512, number_classes)
        self.Auxiliary_Block_2 = Auxiliary_Block(528, number_classes)

    def call(self, input):
        output = self.First_Conv_Block(input)
        output = self.Max_pooling_operation_1st(output)
        output = self.Second_Conv_Block(output)
        output = self.Third_Conv_Block(output)
        output = self.Max_pooling_operation_2nd(output)
        output = self.Inception_Block_1(output)
        output = self.Inception_Block_2(output)
        output = self.Max_pooling_operation_3rd(output)
        output = self.Inception_Block_3(output)

        Auxiliary_Block_1_output = self.Auxiliary_Block_1(output)

        output = self.Inception_Block_4(output)
        output = self.Inception_Block_5(output)
        output = self.Inception_Block_6(output)

        Auxiliary_Block_2_output = self.Auxiliary_Block_2(output)

        output = self.Inception_Block_7(output)
        output = self.Max_pooling_operation_4th(output)
        output = self.Inception_Block_8(output)
        output = self.Inception_Block_9(output)
        output = self.Max_pooling_operation_5th(output)
        output = tf.keras.layers.Flatten(output, 1)
        output = self.dropout(output)
        output = self.fc(output)

        return output, Auxiliary_Block_1_output, Auxiliary_Block_2_output



