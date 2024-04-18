import torch 
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt
import imageio

"""

    Methods to Train the Network 
    Largely Based off of this tutorial: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

"""



def transform_to_numpy(image_grid, epoch):

    image_grid = image_grid.numpy().transpose(1,2,0)
    #means and variances calculates over the training data
    mean = np.array([0.4622019469809971, 0.4670948782918414, 0.43085219394266233])
    std = np.array([0.21187020615982455, 0.23291046662010834, 0.19555240517935008])

    image_grid = std*image_grid + mean

    return image_grid

class Network_Trainer():

    def __init__(self, model,
                 transforms, criterion,
                 epochs, device, optimizer = 'SGD', 
                 batch_size = 10, 
                 learning_rate =.001, out_path = ""):
        
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

        self.out_path = out_path


        self.transform = transforms
        #computation device
        self.device = device
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size

        self.images = []
        

    def train(self, dataloader, train_data):

            print('Training')
            self.model.train()

            train_running_loss = 0.0
            train_running_correct = 0

            for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):

                data, target = data[0].to(self.device), data[1].to(self.device)
                int_targets = torch.argmax(target, 1)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)

                loss = self.criterion(outputs, int_targets)
                train_running_loss += loss.item()

                #print(outputs.data)
                _, preds = torch.max(outputs.data, 1)

                train_running_correct += (preds == int_targets).sum().item()

                loss.backward()
                self.optimizer.step()
                
            train_loss = train_running_loss/len(dataloader.dataset)
            train_accuracy = 100. * train_running_correct/len(dataloader.dataset)    
            return train_loss, train_accuracy


    def stn_grid(self, epoch, val_loader):

        with torch.no_grad():
            data = next(iter(val_loader))[0].to(self.device)

            #print(transformed_image)
            transformed_image = self.model.spt_forward(data).cpu().detach()
            image_grid = torchvision.utils.make_grid(transformed_image)

            image_grid = transform_to_numpy(image_grid, epoch)
            plt.imshow(image_grid)
            plt.savefig(self.out_path + f"/image_{epoch}.png")
            plt.close()
            self.images.append(image_grid)

    
    def stn_initial_grid(self, epoch, val_loader):

        data = next(iter(val_loader))[0].to(self.device)

        #print(transformed_image)
        image_grid = torchvision.utils.make_grid(data)

        image_grid = transform_to_numpy(image_grid, epoch)
        plt.imshow(image_grid)
        plt.savefig(self.out_path + f"/image_{epoch}.png")
        plt.close()
        self.images.append(image_grid)


    def validate(self, dataloader, val_data):
        print('Validating')
        self.model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total = int(len(val_data)/dataloader.batch_size)):
                
                data, target = data[0].to(self.device), data[1].to(self.device)

                int_targets = torch.argmax(target, 1)

                outputs = self.model(data)
                loss = self.criterion(outputs, target)


                val_running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                val_running_correct += (preds == int_targets).sum().item()

        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100.*val_running_correct/len(dataloader.dataset)

        return val_loss, val_accuracy
    

    def train_model(self, train_loader, train_data, val_loader, val_data):

        self.stn_initial_grid(-1, val_loader)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss, train_epoch_accuracy = self.train(train_loader, train_data)
            val_epoch_loss, val_epoch_accuracy =self.validate(val_loader, val_data)

            print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
            print(f"Validation Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}")

            self.stn_grid(epoch, val_loader)

        #imageio.mimsave(self.out_path + '/transformed_imgs.gif', 255*np.array(self.images).astype(np.uint8))










        




        
        























