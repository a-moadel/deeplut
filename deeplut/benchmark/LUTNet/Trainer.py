from deeplut.benchmark.Utils import get_data_set, seed_everything, train, test
from deeplut.benchmark.LUTNet.Models import LFC, CNV
import torch.optim as optim
from deeplut.optim.OptimWrapper import OptimWrapper
from deeplut.benchmark.ModelWrapper import ModelWrapper
import timeit
import torch


def get_model(dataset, k, mask_builder, device):
    model = None
    if dataset == "MNIST":
        model = LFC(k, mask_builder, device)
    else:
        model = CNV(k, mask_builder, device)
    model.to(device)
    model_warpper = ModelWrapper(model)
    return model, model_warpper


def LUTNetTrainer(
    phase_name,
    dataset,
    model_warpper,
    n_epochs,
    lr,
    binary_optim,
    binarization_level,
    input_expanded,
    device,
):
    train_loader, test_loader = get_data_set(dataset)
    seed_everything()

    model = model_warpper._model

    optimizer = OptimWrapper(
        optim.Adam(model.parameters(), lr=lr), BinaryOptim=binary_optim
    )
    model_warpper.set_trainer_paramters(input_expanded, binarization_level)
    best_accuracy = 0
    for epoch in range(n_epochs):
        start = timeit.default_timer()
        _, training_loss = train(model, device, train_loader, optimizer)
        stop = timeit.default_timer()
        training_time = stop - start
        start = timeit.default_timer()
        accuracy, testing_loss = test(model, device, test_loader)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "best_model_{}".format(phase_name))
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


def create_phase(
    phase_name, n_epochs, lr, binary_optim, binarization_level, input_expanded
):
    phase = type("", (), {})()
    phase.phase_name = phase_name
    phase.n_epochs = n_epochs
    phase.lr = lr
    phase.binary_optim = binary_optim
    phase.binarization_level = binarization_level
    phase.input_expanded = input_expanded
    return phase


def multi_phase_training(phases, dataset, model_warpper, device):
    for phase in phases:
        print(":: START PHASE {}".format(phase.phase_name))
        LUTNetTrainer(
            phase.phase_name,
            dataset,
            model_warpper,
            phase.n_epochs,
            phase.lr,
            phase.binary_optim,
            phase.binarization_level,
            phase.input_expanded,
            device,
        )
