from torch.utils.tensorboard import SummaryWriter
from deeplut.benchmark.Utils import get_data_set, seed_everything, train, test
from deeplut.benchmark.ReBNet.Models import LFC, CNV
import torch.optim as optim
from deeplut.optim.OptimWrapper import OptimWrapper


def ReBNetTrainer(dataset, n_epochs, lr, binary_optim, device):
    train_loader, test_loader = get_data_set(dataset)
    seed_everything()

    model = None
    if dataset == "MNIST":
        model = LFC()
    else:
        model = CNV()
    model.to(device)
    optimizer = OptimWrapper(
        optim.Adam(model.parameters(), lr=lr), BinaryOptim=binary_optim
    )

    for epoch in range(1, n_epochs):
        _, training_loss = train(model, device, train_loader, optimizer)
        accuracy, testing_loss = test(model, device, test_loader)
        print(
            ":: Epoch: {:.4f} train avg loss: {:.4f}, test average loss: {:.4f}, acc: ({:.4f}%)\n".format(
                epoch, training_loss, testing_loss, accuracy
            )
        )
