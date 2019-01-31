from torchvision import datasets, transforms, models
import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt    
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator

# Let's use the normalization of ImageNet models so it's easier to compare
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

def show_debug_warning(is_debug):
    if is_debug:
        print("\n-------------------------------------\n            DEBUG MODE\n-------------------------------------\n")

def imshow(img):
    img[0, :, :] = img[0, :, :] * norm_std[0] + norm_mean[0]
    img[1, :, :] = img[1, :, :] * norm_std[1] + norm_mean[1]
    img[2, :, :] = img[2, :, :] * norm_std[2] + norm_mean[2]    
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def show_batch(image_folder, transform = None, shuffle = False):
    # create a loader on the spot 
    batch_size = 16
    if transform is None:
        transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = norm_mean, std = norm_std)])        
    data = datasets.ImageFolder(image_folder.replace('\\','/'),transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                           num_workers=0, shuffle=shuffle)
    # obtain one batch of images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 8))
    # display images
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(data.classes[labels[idx]])
    plt.show()
    
def create_data(image_folder, transform_train, transform_test, batch_size = 64, num_workers = 16, shuffle = True, fivecrop = False):
    # Prepare the datasets. Access the classes with data["train"].classes
    train_data = datasets.ImageFolder(image_folder + "/train/",transform_train)
    valid_data = datasets.ImageFolder(image_folder + "/valid/",transform_test)
    test_data = datasets.ImageFolder(image_folder + "/test/",transform_test)
    data = {"train" : train_data, "valid" : valid_data, "test" : test_data}
    n_classes = len(train_data.classes)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(data["train"], batch_size=batch_size, 
                                               num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    # If using the 5crop test time augmentation num_workers must be set to 0 otherwise we get an error. 
    if fivecrop:
        num_workers = 0
        batch_size = int(np.floor(batch_size/5))
    valid_loader = torch.utils.data.DataLoader(data["valid"], batch_size=batch_size, 
                                               num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data["test"], batch_size=batch_size, 
                                              num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    loaders = {"train" : train_loader, "valid" : valid_loader, "test" : test_loader}        
    return data, loaders, n_classes

def get_samples_per_class(data):
    # Too slow for any practical purposes. Leaving it for a future commit
    samples_per_class = np.zeros(len(data.classes))
    for _, label in data:
        samples_per_class[label] += 1   
    return samples_per_class

def train_epoch(model,train_loader,optimizer,criterion,device):
    train_loss = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move to GPU
        data, target = data.to(device), target.to(device)
        # Set gradients to 0
        optimizer.zero_grad()
        # Get output
        output = model(data)               
        # Calculate loss
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        # Calculate gradients
        loss.backward()
        # Take step
        optimizer.step()    
    train_loss = train_loss / len(train_loader.dataset)
    return model, train_loss
        
def valid_epoch(model,valid_loader,criterion,device,fivecrop):
    ######################    
    # validate the model #
    ######################
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            # move to GPU
            data, target = data.to(device), target.to(device)
            # if we do test time augmentation with 5crop we'll have an extra dimension in our tensor
            if fivecrop == "mean":
                bs, ncrops, c, h, w = data.size()
                output = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
                output = output.view(bs, ncrops, -1).mean(1)
            elif fivecrop == "max":
                bs, ncrops, c, h, w = data.size()
                output = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
                output = output.view(bs, ncrops, -1).max(1)[0]
            else:
                output = model(data)
            ## update the average validation loss
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
    valid_loss = valid_loss / len(valid_loader.dataset) 
    return valid_loss

def train(n_epochs, loaders, model, optimizer, criterion, device, path_model, fivecrop = None, lr_scheduler = None):
    """Trains, validates, and saves the model and other data in a file"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    train_loss = []
    valid_loss = []
    path_state_dict = f"./temp/temp_state_dict_{str(int(np.abs(np.random.randn()) * 1e12))}.pt"
    # Time everything
    time_start = time.time()
    for epoch in range(1, n_epochs+1):
        time_start_epoch = time.time()    
        # Train this epoch
        model, train_loss_epoch = train_epoch(model,loaders["train"],optimizer,criterion,device)
        train_loss.append(train_loss_epoch)   
        # Validate this epoch
        valid_loss_epoch = valid_epoch(model,loaders["valid"],criterion,device,fivecrop)
        # Call the learning rate scheduler if we have one
        if lr_scheduler is not None:
            lr_scheduler.step(valid_loss_epoch)
        valid_loss.append(valid_loss_epoch)  
        # Save if validation loss is the lowest so far
        if valid_loss_epoch <= valid_loss_min:
            torch.save(model.state_dict(), path_state_dict)
            valid_loss_min = valid_loss_epoch 
        # Print epoch statistics
        print('Epoch {} done in {:.2f} seconds. \tTraining Loss: {:.3f} \tValidation Loss: {:.3f}'.format(
            epoch,             
            time.time() - time_start_epoch,
            train_loss_epoch,
            valid_loss_epoch
            ))   
    # Show final statistics    
    print(f"{n_epochs} epochs ready in {(time.time() - time_start):.3f} seconds. Minimum validation loss: {valid_loss_min:.3f}")
    # Load best config
    model.load_state_dict(torch.load(path_state_dict))
    # Save everything to a file
    model_data = {"model": model, "train_loss": train_loss, "valid_loss": valid_loss}
    torch.save(model_data,path_model)
    # Remove the temporary file
    os.remove(path_state_dict)
    
def try_learning_rates(learning_rates,file_names,image_folder,n_epochs,device):
    if not os.path.isdir("./learning_rates/"):
        os.mkdir("./learning_rates/")
    if len(learning_rates) != len(file_names):
        raise Exception("learning rates and file paths have different number of elements")
    for i in range(len(learning_rates)):
        print(f"Trying / loading learning rate {i+1}/{len(learning_rates)}: lr = {learning_rates[i]:.4f}")
        transform_train = transforms.Compose([
                            transforms.Resize((256,256)),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                            transforms.RandomHorizontalFlip(),   
                            transforms.RandomResizedCrop(224, scale=(0.08,1), ratio=(1,1)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean = norm_mean, std = norm_std)])
        transform_test = transforms.Compose([
                            transforms.Resize((224,224)),  
                            transforms.ToTensor(),
                            transforms.Normalize(mean = norm_mean, std = norm_std)])
        data, loaders, n_classes = create_data(image_folder = image_folder, transform_train = transform_train,
                                               transform_test = transform_test, batch_size = 64, num_workers = 8)
        model = Net_Basic(n_classes, depth_1 = 32, fc_size = 512, p_dropout = 0.5, img_input_size = 224)
        train_save_load_model(model,"./learning_rates/" + file_names[i],loaders,learning_rates[i],n_epochs,device)
        
def try_learning_rates_bn(learning_rates,path_list,image_folder,n_epochs,device):
    if len(learning_rates) != len(path_list):
        raise Exception("learning rates and file paths have different number of elements")
    for i in range(len(learning_rates)):
        print(f"Trying learning rate {i+1}/{len(learning_rates)}: lr = {learning_rates[i]:.4f}")
        transform_train = transforms.Compose([
                            transforms.Resize(256),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(224, scale=(0.08,1), ratio=(1,1)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean = norm_mean, std = norm_std)])
        transform_test = transforms.Compose([
                            transforms.Resize(256),  
                            transforms.FiveCrop(224),
                            transforms.Lambda(lambda crops: torch.stack([transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean = norm_mean, std = norm_std)])(crop) for crop in crops]))])
        # We need to decrease the batch size to keep the same memory requirements. This means testing will be about 5x slower.
        data, loaders, n_classes = create_data(image_folder = image_folder, transform_train = transform_train,
                                               transform_test = transform_test, 
                                               batch_size = 64, num_workers = 8, fivecrop = True)
        model = Net_BN(n_classes, depth_1 = 32)
        train_save_load_model(model,path_list[i],loaders,learning_rates[i],n_epochs,device, 
                              fivecrop = "mean", do_lr_scheduling = True)

def load_model_data(path_model):
    model_data = torch.load(path_model)
    model = model_data["model"]
    train_loss = model_data["train_loss"]
    valid_loss = model_data["valid_loss"]
    return model, train_loss, valid_loss

def train_save_load_model(model,path_model,loaders,lr,n_epochs,device, fivecrop = None, do_lr_scheduling = False):
    if os.path.isfile(path_model):
        model, train_loss, valid_loss = load_model_data(path_model)
    else:
        # move tensors to GPU if CUDA is available
        model = model.to(device)
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Optimizer
        optimizer = optim.Adam(model.parameters(),lr) 
        if do_lr_scheduling:
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True)
        else:
            scheduler = None
        # Train
        train(n_epochs, loaders, model, optimizer, criterion, device, path_model, 
              fivecrop = fivecrop, lr_scheduler = scheduler)
        model_data = torch.load(path_model)
        model = model_data["model"]
        train_loss = model_data["train_loss"]
        valid_loss = model_data["valid_loss"]    
    return model, train_loss, valid_loss

def get_losses_for_all_models(path_list):
    all_train_losses = []
    all_valid_losses = []
    all_train_lengths = []
    all_valid_lengths = []
    n = 0
    for path in path_list:
        _, train_loss, valid_loss = load_model_data(path)
        all_train_losses.append(train_loss)
        all_valid_losses.append(valid_loss)
        all_train_lengths.append(len(train_loss))
        all_valid_lengths.append(len(valid_loss))
    if len(set(all_train_lengths + all_valid_lengths)) != 1:
        raise Exception("All training and validation losses should be of the same size")        
    return all_train_losses, all_valid_losses

def show_loss_one_model(train_loss, valid_loss, title = None):
    n_epochs = len(train_loss)
    epochs = np.arange(1,n_epochs+1,1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(epochs, train_loss, label='Train')
    ax.plot(epochs, valid_loss, label='Valid')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.minorticks_on()
    if title is not None:
        plt.title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()    

def show_loss_many_models(path_list, model_names = None):
    all_train_losses, all_valid_losses = get_losses_for_all_models(path_list)       
    n_epochs = len(all_train_losses[0])
    epochs = np.arange(1,n_epochs+1,1)
    fig, ax = plt.subplots(1,2,figsize=(16,4))   
    for i in range(len(all_train_losses)):
        if model_names is not None:
            ax[0].plot(epochs, all_train_losses[i], label=model_names[i])
            ax[1].plot(epochs, all_valid_losses[i], label=model_names[i])
        else:
            ax[0].plot(epochs, all_train_losses[i])
            ax[1].plot(epochs, all_valid_losses[i])
    for i in range(2):
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Loss")
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())  
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation loss")
    if model_names is not None:
        ax[0].legend()
        ax[1].legend()
    plt.show()  

def test(loaders, model, criterion, device):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            data, target = data.to(device), target.to(device)
            bs, ncrops, c, h, w = data.size()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
            output = output.view(bs, ncrops, -1).mean(1)        
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)            
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    
def show_test_batch(model, loader, data, device, n_batches):
    dataiter = iter(loader)
    for i in range(n_batches):
        images, labels = dataiter.next()
        images.numpy()
        # move model inputs to cuda, if GPU available
        images = images.to(device)
        # get sample outputs
        with torch.no_grad():
            if len(images.shape) == 5:
                bs, ncrops, c, h, w = images.size()
                output = model(images.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1) 
            else:
                output = model(data)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        images = images.to("cpu")
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(25, 8))
        for idx in np.arange(output.size(0)):
            ax = fig.add_subplot(2, output.size(0)/2, idx+1, xticks=[], yticks=[])
            imshow(images[idx][-1,:,:,:])
            ax.set_title("{}\n({})".format(data.classes[preds[idx]], data.classes[labels[idx]]),
                         color=("green" if preds[idx]==labels[idx].item() else "red"))
        plt.show()
        
############################################################################################################################################
########################################################### Model architectures ############################################################ 
############################################################################################################################################
# It's not the most elegant solution, but if I would have created an all-in-one master function for network creation I'd have to have first experimented with all architectures before running the code. This way I can experiment on one GPU and run it on the other.

# CNN+FC
class Net_Basic(nn.Module):
    def __init__(self, n_classes, depth_1 = 32, fc_size = 512, p_dropout = 0.5, img_input_size = 224):
        super(Net_Basic, self).__init__()        
        # Keep track of things
        depth_2 = depth_1 * 2
        depth_3 = depth_2 * 2
        self.final_size = img_input_size        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # Conv set 1
        self.conv1_1 = nn.Conv2d(3,depth_1,3,stride = 1,padding = 1)
        self.conv1_2 = nn.Conv2d(depth_1,depth_1,3,stride = 1,padding = 1)
        self.final_size = self.final_size / 2
        # Conv set 2
        self.conv2_1 = nn.Conv2d(depth_1,depth_2,3,stride = 1,padding = 1)
        self.conv2_2 = nn.Conv2d(depth_2,depth_2,3,stride = 1,padding = 1)
        self.final_size = self.final_size / 2
        # Conv set 3
        self.conv3_1 = nn.Conv2d(depth_2,depth_3,3,stride = 1,padding = 1)
        self.conv3_2 = nn.Conv2d(depth_3,depth_3,3,stride = 1,padding = 1)
        self.final_size = self.final_size / 2
        # Input size for fully connected layer
        self.flat_size = int(depth_3 * self.final_size * self.final_size)
        # Fully connected layer
        self.fc6 = nn.Linear(self.flat_size, fc_size)
        self.fc_out = nn.Linear(fc_size,n_classes)
        # Dropout
        self.dropout = nn.Dropout(p = p_dropout)
        
    def forward(self, x):
        # Conv 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        # Conv 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        # Conv 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)
        # Flatten input
        x = x.view(x.size(0), -1)
        # Dropout
        x = self.dropout(x)
        # Fully connected layer
        x = F.relu(self.fc6(x))
        # Dropout
        x = self.dropout(x)
        # Output
        x = self.fc_out(x)

        return x

# CNN + 2FC
class Net_2FC(nn.Module):
    def __init__(self, n_classes, depth_1 = 32, fc_size = 512, p_dropout = 0.5, img_input_size = 224):
        super(Net_2FC, self).__init__()        
        # Keep track of things
        depth_2 = depth_1 * 2
        depth_3 = depth_2 * 2
        self.final_size = img_input_size        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # Conv set 1
        self.conv1_1 = nn.Conv2d(3,depth_1,3,stride = 1,padding = 1)
        self.conv1_2 = nn.Conv2d(depth_1,depth_1,3,stride = 1,padding = 1)
        self.final_size = self.final_size / 2
        # Conv set 2
        self.conv2_1 = nn.Conv2d(depth_1,depth_2,3,stride = 1,padding = 1)
        self.conv2_2 = nn.Conv2d(depth_2,depth_2,3,stride = 1,padding = 1)
        self.final_size = self.final_size / 2
        # Conv set 3
        self.conv3_1 = nn.Conv2d(depth_2,depth_3,3,stride = 1,padding = 1)
        self.conv3_2 = nn.Conv2d(depth_3,depth_3,3,stride = 1,padding = 1)
        self.final_size = self.final_size / 2
        # Input size for fully connected layer
        self.flat_size = int(depth_3 * self.final_size * self.final_size)
        # We add an extra fully connected layer
        self.fc6 = nn.Linear(self.flat_size, fc_size)
        self.fc7 = nn.Linear(fc_size, fc_size)
        self.fc_out = nn.Linear(fc_size,n_classes)
        # Dropout
        self.dropout = nn.Dropout(p = p_dropout)
        
    def forward(self, x):
        # Conv 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        # Conv 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        # Conv 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)
        # Flatten input
        x = x.view(x.size(0), -1)
        # Dropout
        x = self.dropout(x)
        # Fully connected layers
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        # Output
        x = self.fc_out(x)

        return x

# CNN + Pool
class Net_Pool(nn.Module):
    def __init__(self, n_classes, depth_1 = 32, pool_type = 'mean', extra_layer = False, 
                 p_dropout = 0, p_dropout_pool = 0.5, img_input_size = 224, weight_init = False):
        super(Net_Pool, self).__init__()        
        # Keep track of things
        self.pool_type = pool_type
        self.extra_layer = extra_layer
        depth_2 = depth_1 * 2
        depth_3 = depth_2 * 2      
        # Max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # Conv set 1
        self.conv1_1 = nn.Conv2d(3,depth_1,3,stride = 1,padding = 1)
        self.conv1_2 = nn.Conv2d(depth_1,depth_1,3,stride = 1,padding = 1)
        # Conv set 2
        self.conv2_1 = nn.Conv2d(depth_1,depth_2,3,stride = 1,padding = 1)
        self.conv2_2 = nn.Conv2d(depth_2,depth_2,3,stride = 1,padding = 1)
        # Conv set 3
        self.conv3_1 = nn.Conv2d(depth_2,depth_3,3,stride = 1,padding = 1)
        self.conv3_2 = nn.Conv2d(depth_3,depth_3,3,stride = 1,padding = 1)
        self.fc_out = nn.Linear(depth_3,n_classes)
        # Dropout
        self.dropout = nn.Dropout(p = p_dropout)
        # Linear layer to replace global pooling. Input vector will be (mean, std, min, max) x depth_3 size
        self.fc_pool_1 = nn.Linear(depth_3 * 4, n_classes)
        self.fc_pool_2 = nn.Linear(n_classes,n_classes)
        self.fc_pool_out = nn.Linear(n_classes, n_classes)
        self.dropout_pool = nn.Dropout(p = p_dropout_pool)        
        # Initialize weights
        if weight_init:
            nn.init.kaiming_normal_(self.conv1_1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv1_2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2_1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2_2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3_1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3_2.weight, nonlinearity='relu') 
            
    def forward(self, x):
        # Conv 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        # Conv 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        # Conv 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)
        # Pooling, depending on type
        # First we fuse the height and width dimensions (2 and 3) 
        x = x.view(x.size(0),x.size(1),-1)
        if self.pool_type == 'mean':
            single_statistic_pooling = True
            x = x.mean(2)            
        elif self.pool_type == "max":
            single_statistic_pooling = True
            x = x.max(2)[0]
        elif self.pool_type == "thresh":
            single_statistic_pooling = True
            # The following are 3D tensors of size (n_batches, depth_3, height x width)
            mean_repeated = x.mean(2).unsqueeze(2).repeat(1,1,x.size(2))
            std_repeated = x.std(2).unsqueeze(2).repeat(1,1,x.size(2))
            thresh = mean_repeated + 2 * std_repeated     
            mask = (x >= thresh).float()
            x = (mask * x).mean(2)           
        elif self.pool_type == "linear":
            single_statistic_pooling = True
            raise Exception("Not yet implemented")
        elif self.pool_type == "non-linear single":
            single_statistic_pooling = True
            raise Exception("Not yet implemented")            
        elif self.pool_type == "non-linear multi":
            single_statistic_pooling = False
            # Here we concatenate basic statistics about all slices
            x = torch.cat((x.mean(2).unsqueeze(2),x.std(2).unsqueeze(2),
                               x.min(2)[0].unsqueeze(2),x.max(2)[0].unsqueeze(2)), dim = 2).view(x.size(0),-1)                 
        else:
            raise Exception("Choose pool_type between mean, max, thresh, and linear")
        # Output
        if single_statistic_pooling:
            x = self.dropout(x)
            x = self.fc_out(x)
        else:
            x = self.dropout_pool(F.relu(self.fc_pool_1(x)))
            if self.extra_layer:
                x = self.dropout_pool(F.relu(self.fc_pool_2(x)))
            x = self.fc_pool_out(x)
        return x

# CNN + Batchnorm
class Net_BN(nn.Module):
    def __init__(self, n_classes, depth_1 = 32):
        super(Net_BN, self).__init__()        
        # Keep track of things
        depth_2 = depth_1 * 2
        depth_3 = depth_2 * 2      
        # Max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # Conv set 1
        self.conv1_1 = nn.Conv2d(3,depth_1,3,stride = 1,padding = 1)
        self.conv1_2 = nn.Conv2d(depth_1,depth_1,3,stride = 1,padding = 1)
        self.bn1_1 = nn.BatchNorm2d(depth_1)
        self.bn1_2 = nn.BatchNorm2d(depth_1)
        # Conv set 2
        self.conv2_1 = nn.Conv2d(depth_1,depth_2,3,stride = 1,padding = 1)
        self.conv2_2 = nn.Conv2d(depth_2,depth_2,3,stride = 1,padding = 1)
        self.bn2_1 = nn.BatchNorm2d(depth_2)
        self.bn2_2 = nn.BatchNorm2d(depth_2)        
        # Conv set 3
        self.conv3_1 = nn.Conv2d(depth_2,depth_3,3,stride = 1,padding = 1)
        self.conv3_2 = nn.Conv2d(depth_3,depth_3,3,stride = 1,padding = 1)
        self.bn3_1 = nn.BatchNorm2d(depth_3)
        self.bn3_2 = nn.BatchNorm2d(depth_3)
        # Output
        self.fc_out = nn.Linear(depth_3,n_classes)   
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3_2.weight, nonlinearity='relu') 
            
    def forward(self, x):
        # Conv 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        # Conv 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        # Conv 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)
        # First we fuse the height and width dimensions (2 and 3) 
        x = x.view(x.size(0),x.size(1),-1)        
        # And now max global pooling
        x = x.max(2)[0]
        # Output
        x = self.fc_out(x)
        return x

# CNN + Batchnorm + Residual connections
class Net_BN_Res(nn.Module,):
    def __init__(self, n_classes, depth_1 = 32):
        super(Net_BN_Res, self).__init__()        
        # Keep track of things
        depth_2 = depth_1 * 2
        depth_3 = depth_2 * 2      
        # Max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # Conv set 1
        self.conv1_1 = nn.Conv2d(3,depth_1,3,stride = 1,padding = 1)
        self.conv1_2 = nn.Conv2d(depth_1,depth_1,3,stride = 1,padding = 1)
        self.bn1_1 = nn.BatchNorm2d(depth_1)
        self.bn1_2 = nn.BatchNorm2d(depth_1)
        # Conversion from depth_1 to depth_2
        self.conv1x1_2 = nn.Conv2d(depth_1,depth_2,1,stride = 1,padding = 0)
        # Conv set 2
        self.conv2_1 = nn.Conv2d(depth_2,depth_2,3,stride = 1,padding = 1)
        self.conv2_2 = nn.Conv2d(depth_2,depth_2,3,stride = 1,padding = 1)
        self.bn2_1 = nn.BatchNorm2d(depth_2)
        self.bn2_2 = nn.BatchNorm2d(depth_2)    
        # Conversion from depth_2 to depth_3
        self.conv1x1_3 = nn.Conv2d(depth_2,depth_3,1,stride = 1,padding = 0)        
        # Conv set 3
        self.conv3_1 = nn.Conv2d(depth_3,depth_3,3,stride = 1,padding = 1)
        self.conv3_2 = nn.Conv2d(depth_3,depth_3,3,stride = 1,padding = 1)
        self.bn3_1 = nn.BatchNorm2d(depth_3)
        self.bn3_2 = nn.BatchNorm2d(depth_3)
        # Output
        self.fc_out = nn.Linear(depth_3,n_classes)   
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3_2.weight, nonlinearity='relu') 
            
    def forward(self, x):
        # Conv 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        out = self.conv1x1_2(x)
        # Conv 2
        x = F.relu(self.bn2_1(self.conv2_1(out)))
        x = self.bn2_2(self.conv2_2(x))
        x = F.relu(x + out)
        x = self.pool(x)        
        out = self.conv1x1_3(x)
        # Conv 3
        x = F.relu(self.bn3_1(self.conv3_1(out)))
        x = self.bn3_2(self.conv3_2(x))
        x = F.relu(x + out)
        x = self.pool(x)
        # First we fuse the height and width dimensions (2 and 3) 
        x = x.view(x.size(0),x.size(1),-1)        
        # And now max global pooling
        x = x.max(2)[0]
        # Output
        x = self.fc_out(x)
        return x

