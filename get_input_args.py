#
# PROGRAMMER: Susanta pattanayak
# DATE CREATED:  04/18/2019
# REVISED DATE: 04/18/2019
# PURPOSE: Utility function to retrieve all the command line arguments using argparse
#
##
# Imports python modules
import argparse


#
# 
def get_input_args_training():
    """
    Retrieves and parses the command line arguments (for training a model) provided by the user when
    they run the program from a terminal window.
    Command Line Arguments:
      1. data_directory: Basic usage: python train.py data_directory
      2. --save_dir : Set directory to save checkpoints:
      3. --arch: Choose architecture
      4. Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
      5. Use GPU for training: python train.py data_dir --gpu
      6. --debug: to enable debug and get the print statements
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

    parser = argparse.ArgumentParser(description='Input arguments for the training image processing classifier')

    parser.add_argument('data_directory', metavar='N', type=str, nargs='+',
                        help='Data directory for training')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='path to folder of saved checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Pretrained CNN Model architecture to be used. i.e resnet, alexnet, vgg. default is vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='training learning rate. Default value is 0.002')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='hidden units in the classification layer. Default is 1024')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs, default is 3')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def get_input_args_predict():
    """
        Retrieves and parses the command line arguments( to predict a flower image) provided by the user when
        they run the program from a terminal window.
        Command Line Arguments:
          1. image_path: Basic usage: python train.py image_path
          2. --checkpoint : Path of saved trained model
          3. --arch: Choose architecture
          4. --top_k: Return top KK most likely classes
          5. --category_names: Use a mapping of categories to real names:
          6. --debug: pass debug to get debug print statements
          7. Use GPU for training: python train.py data_dir --gpu
        This function returns these arguments as an ArgumentParser object.
        Parameters:
         None - simply using argparse module to create & store command line arguments
        Returns:
         parse_args() -data structure that stores the command line arguments object
        """

    parser = argparse.ArgumentParser(description='Input arguments for the predicting image from a trained model')

    parser.add_argument('image_path', metavar='N', type=str, nargs='+',
                        help='Path of the image file to be predicted')
    parser.add_argument('checkpoint', metavar='N', type=str, nargs='+',
                        help='Saved checkpoint to load the model')

    # Do not put a default value, As if it's not set then you are just going to predict the flower
    # name with the highest probability.
    parser.add_argument('--top_k', type=int, help='Return top KK most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help="Use a mapping of categories to real names: ")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()
