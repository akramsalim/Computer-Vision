''' This is an implementation of a research paper called "Going deeper with convolutions" which discussed
    the idea of Inception V1 Architecture''' 
''' The paper could be found here "https://arxiv.org/abs/1409.4842" '''


import numpy as np
from matplotlib import pyplot as plt
from torch import device
#from torch import device
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

import tensorflow as tf
from tensorflow import keras



tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)


# Building the initial Convolutional Block
class ConvBlock(keras.layers.Layer):
      def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = tf.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = tf.nn.BatchNorm2d(out_channels)
        self.activation = tf.nn.relu()

      def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

"""
## Building the Inception Block

### “#3×3 reduce” and “#5×5 reduce”

From Paper - “#3 × 3 reduce” and “#5 × 5 reduce” stands for the number of 1 × 1 filters
in the reduction layer used before the 3 × 3 and 5 × 5 convolutions.
One can see the number of 1 × 1 filters in the projection layer after the
built-in max-pooling in the “pool proj” column.
All these reduction/ projection layers use rectified linear (ReLU) activation.

"""


class Inception(keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        num1x1,
        num3x3_reduce,
        num3x3,
        num5x5_reduce,
        num5x5,
        pool_proj,
    ):
        super(Inception, self).__init__()

        # Four output channel for each parallel block of network
        # Note, within Inception the individual blocks are running parallely
        # NOT sequentially.
        self.block1 = tf.nn.Sequential(
            ConvBlock(in_channels, num1x1, kernel_size=1, stride=1, padding=0)
        )

        self.block2 = tf.nn.Sequential(
            ConvBlock(in_channels, num3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num3x3_reduce, num3x3, kernel_size=3, stride=1, padding=1),
        )

        self.block3 = tf.nn.Sequential(
            ConvBlock(in_channels, num5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num5x5_reduce, num5x5, kernel_size=5, stride=1, padding=2),
        )

        self.block4 = tf.nn.Sequential(
            tf.nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # Note the different way this forward function
        # calculates the output.
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)

        return tf.keras.layers.Concatenate([block1, block2, block3, block4], 1)



# Building the Auxiliary Block
class Auxiliary(keras.layers.Layer):
    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()

        self.pool = tf.nn.AdaptiveAvgPool2d((4, 4))
        self.conv = tf.nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.activation = tf.nn.ReLU()

        self.fc1 = tf.nn.Linear(2048, 1024)
        self.dropout = tf.nn.Dropout(0.7)
        self.fc2 = tf.nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pool(x)

        out = self.conv(out)
        out = self.activation(out)
        print('out shape is  ', out.shape)
        # out shape is  torch.Size([2, 128, 4, 4])

        out = tf.keras.layers.Flatten(out, 1)

        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out


# Building the whole Network blocks

class GoogLeNet(keras.layers.Layer):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool3 = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception3A = Inception(
            in_channels=192,
            num1x1=64,
            num3x3_reduce=96,
            num3x3=128,
            num5x5_reduce=16,
            num5x5=32,
            pool_proj=32,
        )
        self.inception3B = Inception(
            in_channels=256,
            num1x1=128,
            num3x3_reduce=128,
            num3x3=192,
            num5x5_reduce=32,
            num5x5=96,
            pool_proj=64,
        )
        self.pool4 = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception4A = Inception(
            in_channels=480,
            num1x1=192,
            num3x3_reduce=96,
            num3x3=208,
            num5x5_reduce=16,
            num5x5=48,
            pool_proj=64,
        )
        self.inception4B = Inception(
            in_channels=512,
            num1x1=160,
            num3x3_reduce=112,
            num3x3=224,
            num5x5_reduce=24,
            num5x5=64,
            pool_proj=64,
        )
        self.inception4C = Inception(
            in_channels=512,
            num1x1=128,
            num3x3_reduce=128,
            num3x3=256,
            num5x5_reduce=24,
            num5x5=64,
            pool_proj=64,
        )
        self.inception4D = Inception(
            in_channels=512,
            num1x1=112,
            num3x3_reduce=144,
            num3x3=288,
            num5x5_reduce=32,
            num5x5=64,
            pool_proj=64,
        )
        self.inception4E = Inception(
            in_channels=528,
            num1x1=256,
            num3x3_reduce=160,
            num3x3=320,
            num5x5_reduce=32,
            num5x5=128,
            pool_proj=128,
        )
        self.pool5 = tf.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception5A = Inception(
            in_channels=832,
            num1x1=256,
            num3x3_reduce=160,
            num3x3=320,
            num5x5_reduce=32,
            num5x5=128,
            pool_proj=128,
        )
        self.inception5B = Inception(
            in_channels=832,
            num1x1=384,
            num3x3_reduce=192,
            num3x3=384,
            num5x5_reduce=48,
            num5x5=128,
            pool_proj=128,
        )
        self.pool6 = tf.nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = tf.nn.Dropout(0.4)
        self.fc = tf.nn.Linear(1024, num_classes)

        self.aux4A = Auxiliary(512, num_classes)
        self.aux4D = Auxiliary(528, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.inception3A(out)
        out = self.inception3B(out)
        out = self.pool4(out)
        out = self.inception4A(out)

        aux1 = self.aux4A(out)

        out = self.inception4B(out)
        out = self.inception4C(out)
        out = self.inception4D(out)

        aux2 = self.aux4D(out)

        out = self.inception4E(out)
        out = self.pool5(out)
        out = self.inception5A(out)
        out = self.inception5B(out)
        out = self.pool6(out)
        out = tf.keras.layers.Flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out, aux1, aux2


''' Reason for using `2048` in line `nn.Linear(2048, 1024)` inside the Auxiliary() class's forward() method

### First note the Rule about Linear Layer input-shape calculation inside a CNN — When transitioning from a convolutional layer output to a linear layer input - I must resize Conv Layer output which is a 4d-Tensor to a 2d-Tensor using view.

A. So, a conv output of `[batch_size, num_channel, height, width]` should be “flattened” to become a `[batch_size, num_channel x height x width]` tensor.

B. And the in_`features` of the linear layer should be set to `[num_channel * height * width]`

===========================================================================

So in this case for GoogLeNet with CIFAR-10 Dataset first check the shape of the 4-D Tensor (i.e. the output tensor from the last Conv Layer) before flattening by doing this -

inside the `forward()` method of the `Auxiliary()` class. put a print statement after the last Conv Layer's activation, like below

###

#out = self .activation(out)
#print('out shape is  ', out.shape)
# out shape is  torch.Size([2, 128, 4, 4])
# So that means for the next Linear Layer's
# in_features should be of shape 128 * 4 * 4 i.e. 2048

out = tf.keras.layers.Flatten(out, 1)

###

It would give me `tf.keras.layers.Size([2, 128, 4, 4])` - And so, as per the above rule, for the immediately following Linear Layer's `in_features` would need to be (128 * 4 * 4) which is 2048.

So what I am doing is that, converting or flattening the last conv layer output of [2, 128, 4, 4] shape, which is a 4-D Tensor - to a 2-D Tensor of size

[2, 128 * 4 * 4] tensor. And so the `in_features` of the immediately following linear layer should  be set to [128 * 4 * 4 ] i.e. 2048. '''


def train_model(model, train_loader, val_loader, criterion, optimizer):
    EPOCHS = 15
    train_samples_num = 45000
    val_samples_num = 5000

    train_epoch_loss_history, val_epoch_loss_history = [], []

    for epoch in range(EPOCHS):

        train_running_loss = 0
        correct_train = 0 

        model.train().cuda()

        for inputs, labels in train_loader:
            
            #inputs, labels = torch.input.to(device), labels.to(device)
            device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

            with tf.device(device):
              inputs = tf.convert_to_tensor(inputs, dtype=tf.float32, name='inputs')
              labels = tf.convert_to_tensor(labels, dtype=tf.float32, name='labels')


            """ for every mini-batch during the training phase, we
            typically want to explicitly set the gradients
            to zero before starting to do backpropragation """
            optimizer.zero_grad()

            # Start the forward pass
            prediction0, aux_pred_1, aux_pred_2 = model(inputs)

            # Compute the loss.
            real_loss = criterion(prediction0, labels)
            aux_loss_1 = criterion(aux_pred_1, labels)
            aux_loss_2 = criterion(aux_pred_2, labels)

            loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2

            # do backpropagation and update weights with step()# Backward pass.
            loss.backward()
            optimizer.step()

            # Update the running corrects
            _, predicted = tf.math.maximum(prediction0.data, 1)

            correct_train += (predicted == labels).float().sum().item()

            """ Compute batch loss
            multiply each average batch loss with batch-length.
            The batch-length is inputs.size(0) which gives the number total images in each batch.
            Essentially I am un-averaging the previously calculated Loss """
            train_running_loss += loss.data.item() * inputs.shape[0]

        train_epoch_loss = train_running_loss / train_samples_num

        train_epoch_loss_history.append(train_epoch_loss)

        train_acc = correct_train / train_samples_num

        val_loss = 0
        correct_val = 0

        model.eval().cuda()

        with tf.stop_gradient():
            for inputs, labels in val_loader:
                inputs, labels = tf.keras.Input(device), labels.to(device)

                # Forward pass.
                prediction0, aux_pred_1, aux_pred_2 = model(inputs)

                # Compute the loss.
                real_loss = criterion(prediction0, labels)
                aux_loss_1 = criterion(aux_pred_1, labels)
                aux_loss_2 = criterion(aux_pred_2, labels)

                loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2

                # Compute training accuracy.
                _, predicted = tf.math.maximum(prediction0.data, 1)
                correct_val += (predicted == labels).float().sum().item()

                # Compute batch loss.
                val_loss += loss.data.item() * inputs.shape[0]

            val_loss /= val_samples_num
            val_epoch_loss_history.append(val_loss)
            val_acc = correct_val / val_samples_num

        info = "[For Epoch {}/{}]: train-loss = {:0.5f} | train-acc = {:0.3f} | val-loss = {:0.5f} | val-acc = {:0.3f}"

        print(
            info.format(
                epoch + 1, EPOCHS, train_epoch_loss, train_acc, val_loss, val_acc
            )
        )

        model.save(
            model.state_dict(), "/content/sample_data/checkpoint{}".format(epoch + 1)
        )

    model.save(model.state_dict(), "/content/sample_data/googlenet_model")

    return train_epoch_loss_history, val_epoch_loss_history

#from google.colab import drive
drive.mount('/content/drive')

model = GoogLeNet()


model.to(device)
tf.summary(model, (3, 96, 96))

def cifar_dataloader():
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
            
    # Input Data in Local Machine
    # train_dataset = datasets.CIFAR10('../input_data', train=True, download=True, transform=transform)
    # test_dataset = datasets.CIFAR10('../input_data', train=False, download=True, transform=transform)
    
    # Input Data in Google Drive
    train_dataset = datasets.CIFAR10('/content/drive/MyDrive/All_Datasets/CIFAR10', train=True, download=True, transform=transform)
    
    test_dataset = datasets.CIFAR10('/content/drive/MyDrive/All_Datasets/CIFAR10', train=False, download=True, transform=transform)

    # Split dataset into training set and validation set.
    train_dataset, val_dataset = random_split(train_dataset, (45000, 5000))
    
    print("Image shape of a random sample image : {}".format(train_dataset[0][0].numpy().shape), end = '\n\n')
    
    print("Training Set:   {} images".format(len(train_dataset)))
    print("Validation Set:   {} images".format(len(val_dataset)))
    print("Test Set:       {} images".format(len(test_dataset)))
    
    BATCH_SIZE = 128

    # Generate dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = cifar_dataloader()

criterion = tf.nn.CrossEntropyLoss()
optimizer = tf.keras.optimizers.Adam(model.parameters(), lr=0.001)

train_epoch_loss_history, val_epoch_loss_history = train_model(model, train_loader, val_loader, criterion, optimizer)

model = GoogLeNet()
model.load_state_dict(torch.load('/content/sample_data/googlenet_model'))


num_test_samples = 10000
correct = 0 

model.eval().cuda()

with  tf.stop_gradient():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Make predictions.
        prediction, _, _ = model(inputs)

        # Retrieve predictions indexes.
        _, predicted_class = tf.math.maximum(prediction.data, 1)

        # Compute number of correct predictions.
        correct += (predicted_class == labels).float().sum().item()

test_accuracy = correct / num_test_samples

print('Test accuracy: {}'.format(test_accuracy))
