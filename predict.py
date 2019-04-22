#
# PROGRAMMER: Susanta pattanayak
# DATE CREATED:  04/19/2019
# REVISED DATE: 02/19/2019
# PURPOSE: train.py provides the functions to training a a model with a pretrained model.
#
#
##

# All required importa are here
from get_input_args import get_input_args_predict
from utilities import Utilities
from torchvision import models
from PIL import Image
import numpy as np
import json
import torch

"""
main function for the predict
"""


def main():
    # get all the arguments
    in_arg = get_input_args_predict()

    utils = Utilities(in_arg)
    utils.debug("List of arguments", in_arg)

    image_path = in_arg.image_path[0]

    top_k = 3

    if utils.validate_image_file(image_path) > 0:
        # Proceed with the prediction
        model = load_checkpoint(utils, filepath=in_arg.checkpoint[0])
        device = utils.check_gpu()
        model.to(device)

        try:
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
        except ValueError as e:
            print("JSON object issue: %s") % e

        if in_arg.top_k is not None:
            top_k = in_arg.top_k

        prob, classes, flower_names = predict(image_path, model, in_arg, device, cat_to_name, top_k)
        
        utils.debug("probabilities", prob)
        utils.debug("classes", classes)
        utils.debug("flower_names", flower_names)
        if in_arg.top_k is None:
            # if no top_k is been set just give the predicted flower name
            print("The flower in the image: {} is a {}".format(image_path, flower_names[0]))
        else:
            # if the top_k is been set then give top_k flowers
            print("Top {} most likely classe for the flower image{} are:".format(top_k, image_path))

            for i in range(len(prob)):
                print("Flower: %30s, class: %5s probability: %f" % (flower_names[i], classes[i], prob[i]))


def load_checkpoint(utils, filepath):
    # to avoid: RuntimeError: cuda runtime error (35) :
    # CUDA driver version is insufficient for CUDA runtime version at torch/csrc/cuda/Module.cpp:51
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    model = getattr(models, checkpoint['arch'])(pretrained=True)

    utils.debug("model", model)

    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.input_s = checkpoint['input_s']
    model.output_s = checkpoint['output_s']
    model.epochs = checkpoint['epochs']

    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image_file):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    :param image_file:
    :return:
    """

    # open and resize
    image = Image.open(image_file)
    image = image.resize((256, 256))

    # crop
    left = 0.5 * (image.width - 224)
    bottom = 0.5 * (image.height - 224)
    right = left + 224
    top = bottom + 224
    image = image.crop((left, bottom, right, top))

    # Normalize
    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    image = (image - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))

    # print(image)

    return image


def predict(image_path, model, in_arg, device, cat_to_name, top_k):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    :param image_path:
    :param model:
    :param topk:
    :return:
    """

    # turn off dropout
    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)

    # make sure the image has the right device cpu or cuda
    if device == "cuda":
        image = image.cuda()
    else:
        image = image.cpu()

    image.unsqueeze_(0)

    pos = torch.exp(model.forward(image))

    top_p, top_class = pos.topk(top_k, dim=1)
    top_class_list = top_class.tolist()[0]

    # print(top_class.tolist()[0])
    # print(top_p.tolist()[0])

    # create a flat list of probabilities
    probs = top_p.tolist()[0]
    classes = []
    indices = []
    flower_names = []

    items = model.class_to_idx.items()
    # time to get the classes
    # print(items)
    for k, v in model.class_to_idx.items():
        indices.append(k)
    for i in range(top_k):
        classes.append(indices[top_class_list[i]])
        flower_names.append(cat_to_name.get(indices[top_class_list[i]]))

        # print(flower_names)

    return probs, classes, flower_names


if __name__ == "__main__":
    main()
