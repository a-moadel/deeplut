from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import os
import random
import numpy as np


def get_data_set(dataset):

    if dataset == "SVHN":
        kwargs = {'batch_size': 100}
        transform_train = transforms.Compose([
            transforms.RandomRotation(8),
            transforms.RandomAffine(
                0, translate=(.15, .15), shear=10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.SVHN('../data', split="train", download=True,
                                      transform=transform_train)
        test_dataset = datasets.SVHN('../data', split="test", download=True,
                                     transform=transform_test)
    elif dataset == "MNIST":
        kwargs = {'batch_size': 100}
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True,
                                      transform=transform)
    elif dataset == "CIFAR10":
        kwargs = {'batch_size': 100}
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10('../data', train=False, download=True,
                                        transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, **kwargs, num_workers=3)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, **kwargs, num_workers=3)
    return train_loader, test_loader


def train(model, device, train_loader, optimizer):
    model.train()
    train_losses = []
    running_loss = 0
    batch_count = 0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        running_loss += loss.item()
        loss.backward()
        model.update_grad_expanded()
        optimizer.step()
        train_losses.append(loss.item())
        batch_count += 1
    running_loss /= batch_count
    return train_losses, running_loss


def test(model, device, test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    batch_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_count += 1

    running_loss /= batch_count
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return accuracy, running_loss


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()
