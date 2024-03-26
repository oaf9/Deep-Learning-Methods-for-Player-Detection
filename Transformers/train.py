import torch 
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

"""

    Methods to Train the Network 
    Largely Based off of this tutorial: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

"""

class Network_Trainer():

    def __init__(self, model , data, device, optimizer = 'SGD', batch_size = 1000, epochs= 120,
                 learning_rate =.01, log_steps = 500, file_path = ""):
        
        
        """
        args: 
        model - the model you want to train. 
        device - the device you want to train on. 
        optimizer  - Your optimizer of (default is SGD)
        batch_size - tdesired raining batch size 
        epochs - number of training epochs
        learning_rate - learning rate for training
        log_steps - size of steps for epochs
        file_path the file path where logs should be saved
        """
        
        
        

        self.device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        print(f"Running On {self.device}")

        self.model = model  
        self.device = device
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.path = file_path
        self.dataset = data

        if(optimizer == 'Adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)        
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    

    def train_model(self, epochs ):

        train_loader = DataLoader(self.dataset,
                                  sampler=torch.utils.data.RandomSampler(self.dataset,replacement=True),
                                  shuffle=False,
                                  pin_memory=True,
                                  batch_size=self.batch_size)
        

        running_loss = 0

        for i in range (1, epochs + 1):

            print(f"Epoch: {i}/{epochs}")
            self.model.train()

            for j, data in enumerate(train_loader): 

                inputs, labels = data

                #zero gradients
                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = torch.nn.CrossEntropyLoss()

                loss = loss(outputs, labels)

                #print(outputs)
                #print(inputs)

                running_loss += loss.item()

                loss = loss.backward()
                self.optimizer.step()

                if j == self.batch_size-1:
                    last_loss = running_loss/1000
                    print(f"Epoch {i}: training loss: {last_loss}. Accuracy: {torch.mean((outputs == labels).float())}")
                    running_loss = 0


    def evaluate(self, validation_set):

        v_loader = DataLoader(validation_set,
                                  sampler=torch.utils.data.RandomSampler(self.dataset,replacement=True),
                                  shuffle=False,
                                  pin_memory=True,
                                  batch_size=self.batch_size)


        self.model.eval()
        
        running_loss = 0


        with torch.no_grad():
            for i, data in enumerate(validation_set):

                inputs, labels = validation_set

                #zero gradients
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                loss = loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                last_loss = running_loss/1000

                print(f"epoch {i} validation loss: {last_loss} ]")










        




        
        























