import os as os
from torchvision import models
from PIL import Image
import torch


class Utilities:
    """
    Utilities class provide common validation functions
    """

    def __init__(self, in_args):
        self.in_args = in_args



    def valid_dir(self, dir):
        """
        Utility function to check the valid directory
        :param dir: dir path to validate
        :return: returns 0 if it's not valid directory path. 1 if it is a valid directory path
        """

        is_valid = 1
        if not os.path.isdir(dir):
            print("ERROR: the {} is not a valid directory".format(dir))
            is_valid = 0

        return is_valid

    def valid_arch(self, arch):

        """
        Utility function to validate the architecture
        :param arch: pre-trained model name to be validated
        :return: returns 0 if it's not valid pre-trained  model. 1 if it is a valid model
        """
        # check for valid and allowed archs
        is_valid = 1
        try:
            model = getattr(models, arch)(pretrained=True)
        except:
            print("Error:{} is not a valid  model".format(arch))
            is_valid = 0
        return is_valid

    def debug(self, text, attr):
        """
        A simple debug print system. If --debug is passed through command line. This function gets executed
        :param text: Text to be printed
        :param attr: attribute to be printed
        :return: None
        """
        # only if debug is enabled
        if self.in_args.debug:
            print("#######  Debug #######")
            print(text)
            print(attr)
            print("####### End of Debug #######")

    def validate_image_file(self, file_name):
        """
        Function to validate the image and print the Error results if any
        :param file_name: image filename
        :return: Returns if the image file is valid

        """
        # filename parser should detect and fully ignore any filename starting with '.'
        # Notes: I could have used an if-else. But preferred 'continue'
        is_valid = 1

        if file_name.startswith("."):
            print("\n\nERROR: File {} is not a valid file name as it starts with  '.'".format(file_name))
            is_valid = False
            return is_valid
        # Check if the file has the right extension
        try:
            file_image = Image.open(file_name)
            file_image.verify()
            return is_valid

        except OSError as e:
            # Print the invalid image details
            print('Exception: {} ERROR: File {} is not a valid image \n'.format(e, file_name))
            is_valid = 0
            return is_valid

    def check_gpu(self):
        """
        Setting the gpu option if the user has set
        :return: device: gpu or cpu
        """
        # Find if GPU available, if yes assign the device to GPU

        device = torch.device("cpu")

        # check for the gpu argument else set it up for CPU.
        if self.in_args.gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
        return device
