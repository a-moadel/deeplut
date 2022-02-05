from deeplut.benchmark.Utils import get_data_set, seed_everything, train, test
from deeplut.benchmark.ReBNet.Models import LFC, CNV
import torch.optim as optim
from deeplut.optim.OptimWrapper import OptimWrapper
import timeit
import torch


def ReBNetTrainer(
    phase_name, dataset, n_epochs, lr, binary_training, device, load_path=None
):
    train_loader, test_loader, extra_loader = get_data_set(dataset)
    seed_everything()

    model = None
    if dataset == "MNIST":
        model = LFC(binary_training)
    else:
        model = CNV(binary_training)
    model = torch.nn.DataParallel(model.to(device))
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    best_accuracy = 0
    training_losses = []
    for epoch in range(1,n_epochs+1):
        start = timeit.default_timer()
        _, training_loss = train(model, device, train_loader, optimizer)
        if extra_loader is not None:
            _, extra_loss = train(model, device, extra_loader, optimizer)
            training_loss = (training_loss + extra_loss) / 2
        training_losses.append(training_loss)
        stop = timeit.default_timer()
        training_time = stop - start
        start = timeit.default_timer()
        accuracy, testing_loss = test(model, device, test_loader)
        scheduler.step(testing_loss)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "best_model_{}".format(phase_name))
        if epoch%10==0:
            torch.save(model.state_dict(), "{}_{}".format(epoch, phase_name))
        stop = timeit.default_timer()
        test_time = stop - start
        print(
            ":: Epoch: {} train avg loss: {:.4f} time = {}, test average loss: {:.4f}, acc: ({:.4f}%) time = {}\n".format(
                epoch,
                training_loss,
                training_time,
                testing_loss,
                accuracy,
                test_time,
            )
        )
    return training_losses
