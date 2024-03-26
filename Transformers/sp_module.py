import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


"""
    Repos/papers That Should be Referenced:
    https://github.com/ppriyank/Deep-neural-network-for-traffic-sign-recognition-systems/blob/master/model.py
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    https://arxiv.org/pdf/1506.02025.pdf

"""

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        """
        PRIMARY CLASSIFIER LAYERS
            - Instantiate a Spatial Transformer Network
            - For now, there is a very simple two-layer structure: 
            - Current images have size (3, 40, 100)
        """


        #Convolutional Layers
        self.Conv1 = nn.Conv2d(3, 8, kernel_size = 5, padding = 2)
        self.Conv2 = nn.Conv2d(8, 8, kernel_size = 4, padding = 2)

        #Subsampling Layers
        self.maxPool = nn.MaxPool2d(2,2)

        #Dense Layers
        self.dense_1 = nn.Linear(2000, 50)
        self.dense_2 = nn.Linear(50, 45)

        """ 
        This is the end of the classifier architecture
        """
        



        """ 
        Spacial Transformer Network Module
        """

        self.localizer = nn.Sequential(

            #Convolutional Layers, The Deepmind Paper reccomends this architecture
            nn.Conv2d(3, 16, kernel_size = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
        )

        #Regression Layers: A 2 layer dense neural network
        self.Regress_Theta = nn.Sequential(

            #The input Dimension needs to be calculated from the output of the lcoalization moduel. 
            #otuput of flattened localizer should be 1920, assumng an input shape of (3, 40, 100)
            nn.Linear(4160, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 3*2)
        )




        # Insiitialize weights for Affine tranformations
        
        #As in then paper, Weights are Initilized at the identity. Or in this case, pseudo-identity since the affine 
        # matrix is not square
        # [ 1 0 0 ]
        # [ 0 1 0 ]


        self.Regress_Theta[2].weight.data.zero_()
        self.Regress_Theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.Regress_Theta[2].weight.data.zero_()
        self.Regress_Theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.Regress_Theta[2].weight.data.zero_()
        self.Regress_Theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))



    #forward pass for the spatial Transformer: 
    def spt_forward(self, X):

        #localize the input

        X_loc = self.localizer.forward(X)
        #This resshapes the data so that it is consistent with the input shapes in regression layer
 
        X_loc = X_loc.view(-1, 4160)

        #Get the transformation paramaters
        Θ = self.Regress_Theta(X_loc)

        # make the affine grid (this is where the transformation occurs, then we sample the output)
        grid = F.affine_grid(Θ.view(-1,2,3), X.size())


        output = F.grid_sample(X, grid)



        return output
    

    def forward(self, X):

        #apply the spacial transformer module
        X = self.spt_forward(X)

        #Feed into the classifier. 
        X = self.Conv1(X)
        X = F.relu(X)
        X = self.maxPool(X)

        X = self.Conv2(X)
        X = F.relu(X)
        X = self.maxPool(X)

        X = self.dense_1(X.view(-1, 2000))
        X = F.relu(X)
        X = self.dense_2(X)



        return X