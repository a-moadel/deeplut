import unittest
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from deeplut.nn.Linear import Linear as dLinear
from deeplut.optim import OptimWrapper as dOptimWrapper
from deeplut.mask.MaskMinimal import MaskMinimal
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()

        self.layer1 = dLinear(
            28 * 28,
            128,
            k=2,
            binary_calculations=True,
            trainer_type=LagrangeTrainer,
            mask_builder_type=MaskMinimal,
            bias=False,
        )
        self.activation = torch.nn.ReLU()
        self.final = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.final(x)
        return F.log_softmax(x, dim=-1)


def train(model, device, train_loader, optimizer, epoch):
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
            test_loss /= len(test_loader.dataset)


class Test_TrainMNIST(unittest.TestCase):
    def _test_train_minist(self):
        kwargs = {"batch_size": 64}
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NeuralNetwork().to(device)
        _optim = torch.optim.Adadelta(model.parameters(), lr=0.01)
        optimizer = dOptimWrapper(_optim)
        scheduler = StepLR(_optim, step_size=1)
        for epoch in range(1, 3):
            train(model, device, train_loader, optimizer, epoch)
            evaluate(model, device, test_loader)
            scheduler.step()


if __name__ == "__main__":
    unittest.main()
