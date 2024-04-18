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

        ## MY MODEL ARCHITECTURE
        #Convolutional Layers
        self.Conv1 = nn.Conv2d(3, 8, kernel_size = 5, padding = 2)
        self.Conv2 = nn.Conv2d(8, 8, kernel_size = 4, padding = 2)

        #Subsampling Layers
        self.maxPool = nn.MaxPool2d(2,2)

        #Dense Layers
        #self.dense_1 = nn.Linear(2000, 50)
        self.dense_1 = nn.Linear(392, 50)
        self.dense_2 = nn.Linear(50, 48)
        
        """ 
        This is the end of the classifier architecture

        """

        """ 
        Spacial Transformer Network Module
        """

        self.localizer = nn.Sequential(

            ## MY MODEL ARCHITECTURE

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

            # MY PARAMS ARCHITECTURE
            # The input Dimension needs to be calculated from the output of the lcoalization moduel. 
            # otuput of flattened localizer should be 1920, assumng an input shape of (3, 40, 100)

            nn.Linear(4160, 128),
            nn.ReLU(inplace = True),
             #ORIGINAL OUT
            #nn.Linear(128, 3*2)
            nn.Linear(128, 3)
        )

        # Insiitialize weights for Affine tranformations

        #As in then paper, Weights are Initilized at the identity. Or in this case, pseudo-identity since the affine 
        # matrix is not square
        # [ 1 0 0 ]
        # [ 0 1 0 ]
        self.Regress_Theta[2].weight.data.zero_()
        # self.Regress_Theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.Regress_Theta[2].bias.data.copy_(torch.tensor([0.2, 0, 0], dtype=torch.float))



    #forward pass for the spatial Transformer: 
    def spt_forward(self, X):

        #localize the input

        X_loc = self.localizer(X)
        #This resshapes the data so that it is consistent with the input shapes in regression layer
 
        X_loc = X_loc.view(-1, 4160)

        #Get the transformation paramaters
        Θ = self.Regress_Theta(X_loc)


        translation = Θ[:, 1:].unsqueeze(2)
        scale = Θ[:, 0].unsqueeze(1)
        scale_mat = torch.cat((scale, scale), 1)
        Θ = torch.cat((torch.diag_embed(scale_mat), translation), 2)

        grid = F.affine_grid(Θ, torch.Size([X.size()[0], X.size()[1], 28, 28])) # downsampling from 128x128 to 28x28
        X = F.grid_sample(X, grid)
        
        #ORIGINAL OUT
        # make the affine grid (this is where the transformation occurs, then we sample the output)
        # grid = F.affine_grid(Θ.view(-1,2,3), X.size(), align_corners=True)

        # X = F.grid_sample(X, grid)

        return X
    

    def forward(self, X):

        #my forward call
        # #apply the spacial transformer module
        X = self.spt_forward(X)

        #Feed into the classifier. 
        X = self.Conv1(X)
        X = F.relu(X)
        X = self.maxPool(X)

        X = self.Conv2(X)
        X = F.relu(X)
        X = self.maxPool(X)

        #X = self.dense_1(X.view(-1, 2000))
        X = self.dense_1(X.view(-1, 392))
        X = F.relu(X)
        X = self.dense_2(X)


        return F.log_softmax(X, dim = 1)
