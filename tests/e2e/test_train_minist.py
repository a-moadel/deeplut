import unittest
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from deeplut.nn.Linear import Linear as dLinear
from deeplut.optim.OptimWrapper import OptimWrapper as dOptimWrapper
from deeplut.mask.MaskMinimal import MaskMinimal
from deeplut.mask.MaskExpanded import MaskExpanded
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.layer1 = torch.nn.Linear(28 * 28, 1024)
        self.layer2 = dLinear(
            1024,
            1024,
            k=2,
            binary_calculations=True,
            trainer_type=LagrangeTrainer,
            mask_builder_type=MaskMinimal,
            input_expanded = True,
            bias=False,
        )
        self.layer3 = dLinear(
            1024,
            1024,
            k=2,
            binary_calculations=True,
            trainer_type=LagrangeTrainer,
            mask_builder_type=MaskMinimal,
            input_expanded = True,
            bias=False,
        )
        self.layer2.trainer.set_memorize_as_initializer()
        self.layer3.trainer.set_memorize_as_initializer()
        
        self.activation = torch.nn.ReLU()
        self.final = torch.nn.Linear(1024, 2)

    def forward(self, x, targets: torch.Tensor=None, initalize: bool = False):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x,targets,initalize)
        x = self.activation(x)
        x = self.layer3(x,targets,initalize)
        x = self.final(x)
        return F.log_softmax(x, dim=-1)
    
    def pre_initialize(self):
        self.layer2.pre_initialize()
        self.layer3.pre_initialize()

    def update_initialized_weights(self):
        self.layer2.update_initialized_weights()
        self.layer3.update_initialized_weights()

def intialize(model, device, train_loader):
    model.eval()
    model.pre_initialize()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data, target, True)
    model.update_initialized_weights()

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    zs = 0 
    ones = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.shape[0]
            test_loss /= len(test_loader.dataset)
            zs += torch.sum(target==0)
            ones += torch.sum(target==1)
    tmp = correct/total
    print(tmp)
    print(zs)
    print(ones)
    return correct,total


class Test_TrainMNIST(unittest.TestCase):
    def filter_dataset(self,dataset, classes):
        idx = (dataset.targets == 0) | (dataset.targets == 1)
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]  
        return dataset
    def test_train_minist(self):
        kwargs = {"batch_size": 64}
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
        
        dataset1 = self.filter_dataset(dataset1,[0,1])
        dataset2 = self.filter_dataset(dataset2,[0,1])
        
        train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NeuralNetwork().to(device)
        _optim = torch.optim.Adadelta(model.parameters(), lr=0.01)
        optimizer = dOptimWrapper(_optim)
        scheduler = StepLR(_optim, step_size=1)
        print("Started....")
        evaluate(model, device, test_loader)
        intialize(model,device,train_loader)
        evaluate(model, device, test_loader)
        #for epoch in range(1, 3):
        #    train(model, device, train_loader, optimizer, epoch)
        #    evaluate(model, device, test_loader)
        #    scheduler.step()


if __name__ == "__main__":
    unittest.main()
