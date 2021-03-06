from deeplut.benchmark.Utils import get_data_set, seed_everything, train, test

import torch.optim as optim
import timeit
import torch
from deeplut.nn.BinaryConv2d import BinaryConv2d
from deeplut.nn.BinaryLinear import BinaryLinear


def LUTNetTrainer(
    phase_name,
    dataset,
    model_warpper,
    n_epochs,
    lr,
    binary_training,
    input_expanded,
    device,
):
    train_loader, test_loader, _ = get_data_set(dataset)
    seed_everything()

    model = model_warpper._model

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    if binary_training:
        model_warpper.set_trainer_paramters(input_expanded, 1)
    else:
        model_warpper.set_trainer_paramters(input_expanded, -1)
    for layer in model_warpper._model.layers:
        if isinstance(layer, BinaryLinear) or isinstance(layer, BinaryConv2d):
            layer.set_training_parameters(binary_training)

    best_accuracy = 0
    training_losses = []
    for epoch in range(1, n_epochs + 1):
        start = timeit.default_timer()
        _, training_loss = train(model, device, train_loader, optimizer)
        training_losses.append(training_loss)
        stop = timeit.default_timer()
        training_time = stop - start
        start = timeit.default_timer()
        accuracy, testing_loss = test(model, device, test_loader)
        scheduler.step(testing_loss)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "best_model_{}".format(phase_name))
        if epoch % 10 == 0:
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


def create_phase(
    phase_name,
    n_epochs,
    lr,
    binary_training,
    input_expanded,
    load_path=None,
    use_rebnet=False,
    model_warpper=None,
):
    phase = type("", (), {})()
    phase.phase_name = phase_name
    phase.n_epochs = n_epochs
    phase.lr = lr
    phase.binary_training = binary_training
    phase.input_expanded = input_expanded
    phase.load_path = load_path
    phase.use_rebnet = use_rebnet
    phase.model_warpper = model_warpper
    return phase


def multi_phase_training(phases, dataset, device):
    for phase in phases:
        print(":: START PHASE {}".format(phase.phase_name))
        if phase.load_path is not None:
            load_from_unexpanded_layer(
                phase.model_warpper._model, phase.load_path
            )
        LUTNetTrainer(
            phase.phase_name,
            dataset,
            phase.model_warpper,
            phase.n_epochs,
            phase.lr,
            phase.binary_training,
            phase.input_expanded,
            device,
        )


def load_data(target, src):
    if target.shape == src.shape:
        target.copy_(src)
    else:
        target[:, 0].copy_(src.view(-1))


def load_from_unexpanded_layer(model, path):
    pretrained_dict = torch.load(path)
    new_dict = {}
    for key in pretrained_dict.keys():
        if "module." in key:
            new_dict[key.replace("module.", "")] = pretrained_dict[key]
        else:
            new_dict[key] = pretrained_dict[key]
    pretrained_dict = new_dict
    for key in model.state_dict().keys():
        _new_key = key.replace("trainer.", "")
        if key in pretrained_dict:
            load_data(model.state_dict()[key], pretrained_dict[key])
        else:
            load_data(model.state_dict()[key], pretrained_dict[_new_key])
