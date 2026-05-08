import numpy as np
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import os

import ellipse as el
import swiss as sw

# ==================== I/O OPTIMIZATIONS ====================
# Multi-threaded prefetching configuration
DEFAULT_NUM_WORKERS = 4  # Multi-threaded data loading
PREFETCH_FACTOR = 2     # Number of batches to prefetch
PIN_MEMORY = True       # Pin memory for faster GPU transfer (if GPU available)


class InMemDataLoader:   
    """
    This class is used to download the datasets and converts them into the HDF5 format. 
    The new files will then be used to hold the dataset in memory and reduce the taken to load data 
    during training
    """

    def __init__(self, dataset = 'MNIST', driver = None, root = './data/', conv_sg=False):
        self.driver = driver
        self.dataset = dataset
        self.root = root
        self.conv_sg = conv_sg
        if conv_sg == True:
            self.root = self.root + "conv"   
            #self.dataString = root + "sg" + dataset + ".h5"
        #else:
        self.dataString = root  + dataset + ".h5"     
        
    def loadData(self):
        """
        This loads the dataset and creates the train loader and test loaders
        """
        batch_size = 256
        
        #if self.conv_sg == True:
        #    batch_size = 1        
        
        download = True
        root = self.root + self.dataset
        if self.dataset == "MNIST": 
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            trainset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.MNIST(root, train=False, download=download, transform=transform)
        
        if self.dataset == "CIFAR10":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010,))])
            trainset = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR10(root, train=False, download=download, transform=transform)
        
        if self.dataset == "CIFAR100":
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.CIFAR100(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR100(root, train=False, download=download, transform=transform)
            
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                      shuffle=False, num_workers=0, pin_memory = False)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                             shuffle=False, num_workers=2, pin_memory = False)
        
        return trainloader, testloader
    
    def storeAsHDF5(self):
        
        myFile = h5py.File(self.dataString, 'w', driver= self.driver)
        
        num_classes = 10
        
        if self.dataset == "CIFAR100":
            num_classes = 100           
        
        trainloader, testloader = self.loadData()       
        print("downloading done")
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            #if self.conv_sg == True:
                #tmp = torch.zeros(inputs.shape).float()
                #labels = tmp + labels[0].float()/num_classes
            
            #print(i)
            if i == 0:               
                store_inputs = inputs
                store_labels = labels
                continue
            
            store_inputs = torch.cat((store_inputs, inputs))
            store_labels = torch.cat((store_labels, labels))           
        
        myFile.create_dataset("train_labels", data = store_labels.numpy())    
        myFile.create_dataset("train_inputs", data = store_inputs.numpy(), dtype=np.float)     
        
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            
           # if self.conv_sg == True:
             #   tmp = torch.zeros(inputs.shape).float()
            #    labels = tmp + labels[0].float()/num_classes
            
           # print(i)
            if i == 0:                
                store_inputs = inputs
                store_labels = labels
                continue
            
            store_inputs = torch.cat((store_inputs, inputs))
            store_labels = torch.cat((store_labels, labels))    
            
            
        myFile.create_dataset("test_labels", data = store_labels.numpy())    
        myFile.create_dataset("test_inputs", data = store_inputs.numpy(), dtype=np.float)        
        
        myFile.close()
        
    def getDataset(self, train=True):
        """
        returns the dataset for the given database, for test dataset set train = False
        """
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        # //GPU USED HERE - device selected based on CUDA availability
        
        if self.dataset == "ELLIPSE":
            a = np.array([[0,1.0],[1.0,2.0]])            
            b = a*0.5                            
            myE = el.ellipse(device, 500, 100, a, b)            
            if train == True:
                return myE.create_dataset(myE.examples)
            return myE.create_dataset(myE.valid)                     
    
        if self.dataset == "SWISS":            
            myS = sw.SwissRoll(device, 500, 0.2)            
            if train == True:
                return myS.create_dataset(myS.examples)
            return myS.create_dataset(myS.valid)
                       
               
        #open file
        try:
            myFile = h5py.File(self.dataString, 'r', self.driver)
            
            if train == True:         
                inputString = "train_inputs"
                labelsString = "train_labels"
            
            else:
                inputString = "test_inputs"
                labelsString = "test_labels"
            
            #get hdf5 datsets
            features = myFile.get(inputString)
            labels = myFile.get(labelsString)
           
            # Check if data exists and is valid
            if features is None or labels is None or features.shape[0] == 0:
                raise ValueError("HDF5 file is incomplete or missing data")
            
            #convert to tensors
            features = torch.from_numpy(np.array(features, dtype=np.float32))
            labels = torch.from_numpy(np.array(labels, dtype=np.int64))
            
            #close file to ensure dataset is in memory
            myFile.close()
            
            #conver to correct datatypes
            features = features.float()
            
            if self.conv_sg == False:
                labels = labels.long()       
            dataset = torch.utils.data.TensorDataset(features, labels)
            
            return dataset
        except:
            # Fallback: download from torchvision for MNIST/CIFAR10/CIFAR100
            print(f"HDF5 file incomplete, downloading {self.dataset} from torchvision...")
            if self.dataset == "MNIST":
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                dataset = torchvision.datasets.MNIST('./data/MNIST', train=train, download=True, transform=transform)
                return dataset
            elif self.dataset in ["CIFAR10", "CIFAR100"]:
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                if self.dataset == "CIFAR10":
                    dataset = torchvision.datasets.CIFAR10('./data/CIFAR10', train=train, download=True, transform=transform)
                else:
                    dataset = torchvision.datasets.CIFAR100('./data/CIFAR100', train=train, download=True, transform=transform)
                return dataset
            else:
                raise
        
    
    def getDataLoader(self, batch_size=64, shuffle=True, num_workers=None, pin_memory=None, train=True):
        """
        Optimized DataLoader with multi-threaded prefetching.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker threads (defaults to DEFAULT_NUM_WORKERS for optimization)
            pin_memory: Pin memory for GPU transfer (defaults to PIN_MEMORY if CUDA available)
            train: Whether loading training or test set
        """
        # Use optimized defaults
        if num_workers is None:
            # //GPU USED HERE - use multi-worker dataloading when CUDA is available
            num_workers = DEFAULT_NUM_WORKERS if torch.cuda.is_available() else 0
        
        if pin_memory is None:
            # //GPU USED HERE - pin memory for faster GPU transfer when CUDA available
            pin_memory = PIN_MEMORY and torch.cuda.is_available()
        
        dataset = self.getDataset(train)
        
        # Create DataLoader with prefetching
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=pin_memory,
            prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
            persistent_workers=(num_workers > 0)  # Keep workers alive between epochs
        )
        
        return dataloader
            

def getDims(dataset):
    
    in_channels = 1
    
    if dataset == "ELLIPSE":
        num_classes = 2
        num_features = 2        
   
    if dataset == "SWISS":
        num_classes = 2
        num_features = 4
        
    if dataset == "MNIST":
        num_classes = 10
        num_features = 784
        
    if dataset == "CIFAR10":
        num_classes = 10
        num_features = 1024 
        in_channels = 3
        
    if dataset == "CIFAR100":
        num_classes = 100
        num_features = 1024  
        in_channels = 3
   
    return num_features, num_classes, in_channels



"""
To do 
    add support for labels for convnets
    add support for transforms

"""