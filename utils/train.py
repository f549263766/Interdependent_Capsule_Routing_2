"""
@author: QYZ
@time: 2021/11/26
@file: train.py
@describe: THis file aims to design a trainer function.
"""
import os
import shutil
import gc
import time
import adabound

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

from utils.tools import colorstr
from utils.loss_function import total_loss


class Train:
    def __init__(self, configs, model, data_loader, device):
        # Load variable __init__ method
        self.configs = configs
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = data_loader[0], data_loader[1], data_loader[2]
        self.device = device

        # Setting up the basic training configuration
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=configs.learning_rate,
                                    weight_decay=configs.weight_decay)
        # self.optimizer = optim.SGD(model.parameters(),
        #                            lr=configs.learning_rate,
        #                            momentum=0.95)
        # self.optimizer = adabound.AdaBound(model.parameters(),
        #                                    lr=configs.learning_rate,
        #                                    final_lr=0.1,
        #                                    weight_decay=configs.weight_decay)

        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
        #                                                   gamma=0.95)

        # Initialize variables
        self.start_epoch = 0
        self.best_valid_acc = 0
        self.patience_counter = 0
        self.global_step = 0
        self.best_test_acc = 0

    def train(self):
        # Whether need to load the most recent checkpoint
        if self.configs.resume:
            print(colorstr("[*] Resume training..."))
            self.load_checkpoint(best=False)

        # Start training model on samples
        for epoch in range(self.start_epoch, self.configs.epochs):
            # Logging current train epoch information
            print(colorstr('[*] Current training epoch: ')
                  + colorstr("yellow", '{}'.format(epoch + 1))
                  + "/"
                  + colorstr("magenta", '{}'.format(self.configs.epochs)))
            # Prevent print and tqdm conflict
            time.sleep(0.1)
            # record running time
            start_time = time.time()
            # Miscellaneous settings
            train_sample_count = 0
            train_batch_loss = 0
            train_batch_acc = 0
            # Turn on model training mode
            self.model.train()

            # Use tqdm to decorate the training process
            with tqdm(total=self.train_loader.dataset.__len__()) as pbar:
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    # Loading the data put on the device
                    data = data.to(self.device)
                    target = target.type(torch.LongTensor)
                    target = target.to(self.device)
                    self.global_step = batch_idx + (epoch * len(self.train_loader)) - len(self.train_loader)
                    # Reset the gradient of model
                    self.optimizer.zero_grad()
                    # Getting the output of model
                    output = self.model(data)
                    # Compute the loss of model
                    losses = F.cross_entropy(output, target)
                    # losses = total_loss(output, target, self.global_step, self.device)

                    # Back propagation the loss
                    losses.backward()
                    # Update the gradient
                    self.optimizer.step()

                    # Log the loss and accuracy of each batch
                    train_sample_count += data.size(0)
                    train_batch_loss += losses.item() * data.size(0)
                    train_batch_acc += (output.argmax(-1) == target).sum().item()

                    # Update the tqdm bar
                    pbar.set_description(
                        (
                            "Epoch:{}/{} -- Loss: {:.3f}".format(
                                epoch + 1, self.configs.epochs, losses.data.item()
                            )
                        )
                    )
                    pbar.update(data.shape[0])

            # Prevent print and tqdm conflict
            time.sleep(0.1)
            # Log the loss and accuracy of each epoch
            epoch_train_loss = train_batch_loss / train_sample_count
            epoch_train_acc = train_batch_acc / train_sample_count
            # Wandb logging: Log the scalar values
            wandb.log({
                "{} Train {} Epoch Loss".format(self.configs.model, self.configs.data_name):
                    epoch_train_loss,
                "{} Train {} Epoch Accuracy".format(self.configs.model, self.configs.data_name):
                    epoch_train_acc
            })
            # Measure elapsed time on each epoch
            end_time = time.time()
            print(colorstr('[*] Time elapsed for epoch ')
                  + colorstr("yellow", '{}'.format(epoch + 1))
                  + ": "
                  + colorstr("magenta", '{:.0f}s.'.format(end_time - start_time)))

            # Evaluate on validation set
            with torch.no_grad():
                valid_loss, valid_acc = self.validate(epoch)
            print(colorstr('[*] Val  loss: ')
                  + colorstr("red", '{:.3f}'.format(valid_loss))
                  + " - "
                  + colorstr("Val  acc: ")
                  + colorstr("red", '{:.3f}'.format(valid_acc)))

            # Save the best model
            if valid_acc > self.best_valid_acc:
                self.patience_counter = 0
                is_best = valid_acc > self.best_valid_acc
                self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            else:
                self.patience_counter += 1
                is_best = False
                if self.patience_counter == self.configs.patience or self.best_valid_acc == 100.:
                    print(colorstr('[*] Early stopping... no improvement after {} Epochs.'.format(
                        self.configs.patience)))
                    # release unreferenced memory
                    gc.collect()
                    break

            print(colorstr('[*] The best validation accuracy in currently is : ')
                  + colorstr("red", '{:.3f}'.format(self.best_valid_acc))
                  + colorstr("%"))

            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'patience_counter': self.patience_counter,
                 'best_valid_acc': self.best_valid_acc,
                 'global_step': self.global_step
                 }, is_best
            )
            # release unreferenced memory
            gc.collect()

            # Determine whether to test the model
            if is_best:
                self.test()

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        Args:
            epoch: the current epoch in training.
        """
        # Logging current validation epoch information
        print(colorstr('[*] Validation epoch: ')
              + colorstr("yellow", '{}'.format(epoch + 1))
              + "/ "
              + colorstr("magenta", '{}'.format(self.configs.epochs)))
        # Model evaluate
        self.model.eval()
        # Initialize the args
        correct = 0
        losses = 0
        num_batches = len(self.val_loader)

        # Starting evaluate
        for batch_idx, (data, target) in enumerate(self.val_loader):
            # Loading the data put on the device
            data = data.to(self.device)
            target = target.type(torch.LongTensor)
            target = target.to(self.device)

            # Compute the output of model
            output = self.model(data)
            # Compute the loss of model
            losses += F.cross_entropy(output, target)
            # losses += total_loss(output, target, self.global_step, self.device)
            # Every capsule's L2 function as a probability of exist entity
            pred = output.data.max(1, keepdim=True)[1].to(self.device)
            correct += pred.eq(target.view_as(pred)).sum()

        # Compute the average the loss and accuracy
        losses /= num_batches
        num_test_data = len(self.val_loader.dataset)
        accuracy = torch.true_divide(correct, num_test_data)
        accuracy_percentage = 100. * accuracy

        # Wandb logging: Log the scalar values
        wandb.log({
            "{} Validation {} Accuracy".format(self.configs.model, self.configs.data_name):
                accuracy_percentage,
            "{} Validation {} Total Loss".format(self.configs.model, self.configs.data_name):
                losses
        })

        return losses, accuracy_percentage

    def test(self, best=True):
        """
        Test the model on the test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        # Initialize the args
        correct = 0

        # Load the best checkpoint
        self.load_checkpoint(best=best)
        self.model.eval()

        with torch.no_grad():
            # Starting test the model
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # Loading the data put on the device
                data = data.to(self.device)
                target = target.type(torch.LongTensor)
                target = target.to(self.device)

                # Compute the output of model
                output = self.model(data)
                # Compute the loss of model
                pred = output.data.max(1, keepdim=True)[1].to(self.device)
                correct += pred.eq(target.view_as(pred)).sum()

            perc = (100. * correct) / (len(self.test_loader.dataset))
            error = 100 - perc

            if perc > self.best_test_acc:
                self.best_test_acc = perc
                filename = self.configs.exp_name + "/" + self.configs.model + '_model_best.pth.tar'
                test_filename = self.configs.exp_name + "/" + self.configs.model + '_model_test_best.pth.tar'
                shutil.copyfile(
                    os.path.join("./checkpoints/", filename), os.path.join("./checkpoints/", test_filename)
                )

            if best:
                print(colorstr('[*] Test best Acc: ')
                      + colorstr('yellow', '{}'.format(correct))
                      + colorstr('/')
                      + colorstr("green", "{}".format(len(self.test_loader.dataset)))
                      + colorstr("(")
                      + colorstr("red", "{:.2f}%".format(perc))
                      + colorstr(" - ")
                      + colorstr("magenta", "{:.2f}%".format(error))
                      + colorstr(")"))
            else:
                print(colorstr('[*] Test current Acc: ')
                      + colorstr('yellow', '{}'.format(correct))
                      + colorstr('/')
                      + colorstr("green", "{}".format(len(self.test_loader.dataset)))
                      + colorstr("(")
                      + colorstr("red", "{:.2f}%".format(perc))
                      + colorstr(" - ")
                      + colorstr("magenta", "{:.2f}%".format(error))
                      + colorstr(")"))
            wandb.run.summary["{}_{}_test_accuracy".format(self.configs.model, self.configs.data_name)] = perc
            wandb.run.summary["{}_{}_test_error".format(self.configs.model, self.configs.data_name)] = error
            wandb.log({
                "{} Test {} Accuracy".format(self.configs.model, self.configs.data_name):
                    perc
            })

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.
        Args:
            best: if set to True, loads the best model. Use this if you want
            to evaluate your model on the test data. Else, set to False in
            which case the most recent version of the checkpoint is used.
        """
        print(colorstr("[*] Loading model from ")
              + colorstr("green", "{}".format("./checkpoints")))

        filename = self.configs.exp_name + "/" + self.configs.model + '_ckpt.pth.tar'
        if best:
            filename = self.configs.exp_name + "/" + self.configs.model + '_model_best.pth.tar'
        ckpt_path = os.path.join("./checkpoints/", filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.patience_counter = ckpt['patience_counter']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.global_step = ckpt['global_step']

        if best:
            print(colorstr("[*] Loaded ")
                  + colorstr("magenta", "{} ".format(filename))
                  + colorstr("checkpoint @ epoch ")
                  + colorstr("yellow", "{}".format(ckpt['epoch']))
                  + colorstr(" with best valid acc of ")
                  + colorstr("red", "{:.3f}".format(ckpt['best_valid_acc']))
                  )
        else:
            print(colorstr("[*] Loaded ")
                  + colorstr("magenta", "{} ".format(filename))
                  + colorstr("checkpoint @ epoch ")
                  + colorstr("yellow", "{}".format(ckpt['epoch']))
                  )

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.
        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.configs.exp_name
        ckpt_path = os.path.join("./checkpoints/", filename)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        ckpt_path = os.path.join(ckpt_path, self.configs.model + '_ckpt.pth.tar')
        torch.save(state, ckpt_path)

        if is_best:
            # Logging current best validation accuracy
            print(colorstr('[*] Saving the best validation accuracy model in epoch : ')
                  + colorstr("yellow", '{}'.format(state['epoch'] + 1))
                  + colorstr(" and update the accuracy is ")
                  + colorstr("red", '{}'.format(state['best_valid_acc'])))
            filename = self.configs.exp_name + "/" + self.configs.model + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join("./checkpoints/", filename)
            )
