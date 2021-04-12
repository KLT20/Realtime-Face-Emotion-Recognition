#define the architecture of the cnn, localization function and network forward function

#following @omarsayed7

#complimentary pytorch tutorial found at https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

#channels numbers understood through the following Youtube Video: https://www.youtube.com/watch?v=pDdP0TFzsoQ&ab_channel=PythonEngineer

import torch
import torch.nn as nn
import torch.nn.functional as F

class Emotion(nn.Module):
    
    def __init__(self):
        
        super(Emotion,self).__init__()
        
        self.norm = nn.BatchNorm2d(10)
        
        #in the first layer, the input channel is 1 because FER2013 images are greyscale
        self.layer1 = nn.Conv2d(1,10,3)
        #in the second input layer, the input channel needs to be equal to the previous channel's output channels
        self.layer2 = nn.Conv2d(10,10,3)
        # kernel size defines the 3x3 kernel matrix used to do the convolution
        self.layer3 = nn.Conv2d(10,10,3)
        self.layer4 = nn.Conv2d(10,10,3)
        
        #in the case of Max Pooling, the kernel size is 2 and we shift to the right by 2 pixels at each iteration, thus stride is 2
        self.max_val1 = nn.MaxPool2d(2,2)
        self.max_val2 = nn.MaxPool2d(2,2)
        
        # here, we are setting up the fully connected layers
        # the input of nn.Linear is the size of the image after it has passed by all the conv layers,
        # thus 810  since 10*9*9 = 810 (10 is the number of output channels and 9*9 is the size of the image)
        self.function1 = nn.Linear(810,50)
        
        # since we have 7 different classes (types of emotions), our output channel is 7
        self.function2 = nn.Linear(50,7)
        
        #localization network (Spatial transformer) in DeepEmotion2019 paper
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        #the linear transformation at the end of the localization network
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        #Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        transformed_image = self.stn(input)

        transformed_image = F.relu(self.layer1(out))
        transformed_image = self.layer2(out)
        transformed_image = F.relu(self.max_val1(out))

        transformed_image = F.relu(self.layer3(transformed_image))
        transformed_image = self.norm(self.layer4(transformed_image))
        transformed_image = F.relu(self.max_val2(transformed_image))

        transformed_image = F.dropout(transformed_image)
        transformed_image = transformed_image.view(-1, 810)
        transformed_image = F.relu(self.function1(transformed_image))
        transformed_image = self.function2(transformed_image)

        return transformed_image
