#
# PROGRAMMER: Susanta pattanayak
# DATE CREATED:  04/19/2019
# REVISED DATE: 02/19/2019
# PURPOSE: train.py provides the functions to training a a model with a pretrained model.
#
#
##

# All required importa are here
from get_input_args import get_input_args_training
from utilities import Utilities
from os import path
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import time
import torch


"""
main function for the training
"""


def main():
    # get all the arguments
    in_arg = get_input_args_training()
    data_dir = in_arg.data_directory[0]

    utils = Utilities(in_arg)

    utils.debug("List of arguments", in_arg)

    # arg_validation returns a list 1s and 0s. If any validation failed, then the code doesnot execute
    if 0 not in arg_validation(in_arg, utils):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        pretrained_model = in_arg.arch

        # Define your transforms for the training, validation, and testing sets

        train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

        test_data_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])

        validation_data_transforms = transforms.Compose([transforms.Resize(255),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                              [0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder

        train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
        test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)
        validation_image_datasets = datasets.ImageFolder(valid_dir, transform=validation_data_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
        validationloader = torch.utils.data.DataLoader(validation_image_datasets, batch_size=64)

        model = getattr(models, pretrained_model)(pretrained=True)
        utils.debug("Model", model)

        model, criterion, optimizer, device = initialize_model(model, in_arg, utils)

        # Train the model
        training(model, device, trainloader, optimizer, criterion, validationloader, epochs=in_arg.epochs)

        # test the model
        test_model(testloader, device, model, criterion)

        # Save the Checkpoint
        save_checkpoint(model, train_image_datasets, in_arg, utils)


def arg_validation(in_arg, utils):
    """
    Argument validation function
    :param in_arg: takes the command line arguments
    :return: returns a list of results. 0s and 1s(0 = False and 1 = True)
    """
    results = []
    # validate the data directory check whether it's a valid directory
    results.append(utils.valid_dir(in_arg.data_directory[0]))
    results.append(utils.valid_dir(in_arg.save_dir))
    results.append(utils.valid_arch(in_arg.arch))

    return results


def initialize_model(model, in_arg, utils):
    """
    Initializing the model
    :param model: training model
    :return: model,criterion and optimizer
    """
    # Find if GPU available, if yes assign the device to GPU

    device = torch.device("cpu")

    # check for the gpu argument else set it up for CPU.
    if in_arg.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
    utils.debug("set GPU or CPU", device)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Get the input features from the model
    input_feature = list(model.classifier.children())[0].in_features

    model.classifier = nn.Sequential(nn.Linear(input_feature, in_arg.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_arg.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(device)

    return model, criterion, optimizer, device


def training(model, device, trainloader, optimizer, criterion, validationloader, epochs=3, ):
    """
    Trains the model with pretrained model and prints training and validation results
    :param model:
    :param device:
    :param trainloader:
    :param optimizer:
    :param criterion:
    :param validationloader:
    :param epochs:
    """
    # set the epochs, hit and trial to see what would be the best number

    steps = 0
    running_loss = 0
    test_loss = 0
    accuracy = 0

    # track how much time the training took
    start = time.time()
    print('Training initialized')

    for epoch in range(epochs):
        # Training
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device, could be GPU or CPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Training
            # clearing the gradients in the training loop
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation

        model.eval()
        with torch.no_grad():
            for inputs, labels in validationloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # printing the results

        print(f"Epoch {epoch + 1}/{epochs}.. "f"Train loss: {running_loss / steps:.3f}.. "
              f"Validation loss: {test_loss / len(validationloader):.3f}.. "
              f"Validation accuracy: {accuracy / len(validationloader):.3f}")

        # reinitialize
        running_loss = 0
        model.train()
        steps = 0
        test_loss = 0
        accuracy = 0

    print(f" Device = {device};The total Time per training: {(time.time() - start) // 60:.3f}.."
          f"minutes and {(time.time() - start) % 60:.3f} seconds")


def test_model(testloader, device, model, criterion):
    """
    Once we trained the model, it's time to test the model
    :param testloader:
    :param device:
    :param model:
    :param criterion:
    :return:
    """
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Validation loss: {test_loss / len(testloader):.3f}.. "
          f"Validation accuracy: {accuracy / len(testloader):.3f}")


def save_checkpoint(model, train_image_datasets, in_arg, utils):
    """
    save the checkpoint of a trained model
    :param model:
    :param train_image_datasets:
    :param in_arg:
    :param utils:
    :return:
    """

    model.class_to_idx = train_image_datasets.class_to_idx
    checkpoint_params = {'arch': in_arg.arch,
                         'state_dict': model.state_dict(),
                         'class_to_idx': model.class_to_idx,
                         'classifier': model.classifier,
                         'input_s': 25088,
                         'output_s': 102,
                         'epochs': in_arg.epochs}

    checkpoint_file_name = path.join(in_arg.save_dir, 'classifier'+'_'+in_arg.arch+'.pth')

    torch.save(checkpoint_params, checkpoint_file_name)


if __name__ == "__main__":
    main()
