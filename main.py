"""
@author: QYZ
@time: 2021/09/02
@file: main.py
@describe: This file is main function.
"""
import argparse
import time
import os

import torch
import wandb
from torch.backends import cudnn

from utils.load_data import LoadData
from utils.model_select import model_selected
from utils.tools import clearing_unnecessary_parameters, setting_seed, colorstr, tab_print, show_process, model_summary
from utils.train import Train

if __name__ == "__main__":
    """
    +--------------------------------------------------------------------------------------------+
    |                STEP-1: INPUT THE REQUIRED PARAMETERS FOR THE PROJECT                       |
    +--------------------------------------------------------------------------------------------+
    """
    # Command line parameter input
    project_name = "Capsule_Networks"
    # Initialize the setting of cuda
    cudnn.enabled = True
    cudnn.benchmark = True
    # Setting suitable device
    torch.cuda.set_device(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Setting command line parameters
    parser = argparse.ArgumentParser(project_name)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch Size refers to the number of training samples in each Batch")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Setting the number of model training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Setting learning rate for this model")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Optimiser weight decay value")
    parser.add_argument("--resume", type=bool, default=False,
                        help="Whether to resume training")
    parser.add_argument("--resume_name", type=str, default='20211201_1407',
                        help="The name of resume training")
    parser.add_argument('--patience', type=int, default=300,
                        help="Setting the early stop")
    parser.add_argument("--model", type=str, default="CLA_Routing",
                        help="Selective routing model, such as [VB_Routing, CLA_Routing]")
    parser.add_argument("--data_name", type=str, default="smallNORB",
                        help="Selecting the dataset for model")
    parser.add_argument("--data_path", type=str, default="./datasets/",
                        help="Setting the directory path to find the data")
    parser.add_argument("--in_weight", type=int, default=32,
                        help="Enter the width of the image size")
    parser.add_argument("--in_height", type=int, default=32,
                        help="Enter the width of the image size")
    parser.add_argument("--in_channels", type=int, default=2,
                        help="Enter the width of the image size")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="The number of model recognition categories")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Setting the number of workers to load the dataset")
    parser.add_argument("--seed", type=int, default=999,
                        help="Setting the global random seed")
    parser.add_argument("--affNIST_mode", type=str, default=None,
                        help="Setting the mode for generating affNIST dataset, (all, single)")
    parser.add_argument("--smallNORB_mode", type=str, default="classification",
                        help="Setting the mode for generating smallNORB dataset, (azimuth, elevation, classification)")
    parser.add_argument("--n_multi", type=int, default=None,
                        help="Setting the number of generating MultiMNIST dataset in each digits, such as 1000")
    """
    +--------------------------------------------------------------------------------------------+
    |                           STEP-2: LOAD THE CORRESPONDING MODEL                             |
    +--------------------------------------------------------------------------------------------+
    """
    # Select model
    model, args = model_selected(parser, device)
    # Setting the model save path
    if args.resume:
        args.exp_name = os.path.join(args.model,
                                     args.data_name,
                                     args.resume_name)
    else:
        args.exp_name = os.path.join(args.model,
                                     args.data_name,
                                     time.strftime("%Y%m%d_%H%M", time.localtime()))
    if not os.path.exists("./checkpoints/" + args.exp_name):
        os.makedirs("./checkpoints/" + args.exp_name)
    # Clearing Unnecessary Parameters
    args = clearing_unnecessary_parameters(args)
    # Show the current process， step-1
    show_process(step="STEP-1:", process=" INPUT THE REQUIRED PARAMETERS FOR THE PROJECT")
    # logging the important information
    print(colorstr("[*] The project of this operation is ") + colorstr("red", "bold", project_name))
    print(colorstr("[*] The device in use is ") + colorstr("yellow", "bold", device))
    print(colorstr("[*] Current project operating parameter settings"))
    tab_print(configs=vars(args))
    # Show the current process， step-2
    show_process(step="STEP-2:", process=" LOAD THE CORRESPONDING MODEL")
    # Summary the selected model
    model_summary(model, (args.in_channels, args.in_weight, args.in_height), args.batch_size, "cuda")
    # Initialize wandb with parameter Settings
    try:
        name = "{}_{}_{}".format(args.model, args.data_name, args.smallNORB_mode)
    except AttributeError:
        name = "{}_{}".format(args.model, args.data_name)
    wandb.init(project=project_name, name=name, entity="qyz", config=vars(args))
    # Setting the random seed for all random function.
    setting_seed(seed=args.seed)
    """
    +--------------------------------------------------------------------------------------------+
    |                           STEP-3: LOAD THE CORRESPONDING DATASET                           |
    +--------------------------------------------------------------------------------------------+
    """
    # Show the current process， step-3
    show_process(step="STEP-3:", process=" LOAD THE CORRESPONDING DATASET")
    # Setting the loaded data set
    Train_iteration, Valid_iteration, Test_iteration = LoadData(args).select_data()
    print(colorstr("[*] The data set used for this training is: ") + colorstr("red", args.data_name))
    print(colorstr("[*] Number of training examples: ") + colorstr("red", Train_iteration.dataset.__len__()))
    print(colorstr("[*] Number of validation examples: ") + colorstr("red", Valid_iteration.dataset.__len__()))
    print(colorstr("[*] Number of testing examples: ") + colorstr("red", Test_iteration.dataset.__len__()))
    """
    +--------------------------------------------------------------------------------------------+
    |                           STEP-4: START TRAINING THE MODEL                                 |
    +--------------------------------------------------------------------------------------------+
    """
    # Show the current process， step-4
    show_process(step="STEP-4:", process=" START TRAINING THE MODEL")
    # Initialising the trainer
    trainer = Train(configs=args,
                    model=model,
                    data_loader=(Train_iteration, Valid_iteration, Test_iteration),
                    device=device)
    # Start training the model
    trainer.train()
    # Start testing the model
    trainer.test()
    # Finish training and testing
    wandb.run.finish()
