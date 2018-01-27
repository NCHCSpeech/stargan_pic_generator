#this file is a modification of StarGAN solver.py (https://github.com/yunjey/StarGAN/commit/df7ca56de37c2b7a8b9bfbf8937bb9b83f155e97)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import random
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
from PIL import Image


class Inference(object):
    def __init__(self, celebA_loader, config):
        # Data loader
        self.celebA_loader = celebA_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        if self.dataset == 'Both':
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
        else:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x = (x >= 0.5).float()
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
    
    def make_celeb_labels(self, real_c, input_label):
        #Generate domain labels for CelebA for debugging/testing.

        #if dataset == 'CelebA':
        #    return single and multiple attribute changes
        #elif dataset == 'Both':
        #    return single attribute changes
        
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []
        
        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i < 3:
                    c[:3] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1   # opposite value
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        
        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        if self.dataset == 'CelebA':
            for i in range(4):
                fixed_c = real_c.clone()
                for c in fixed_c:
                    if i in [0, 1, 3]:   # Hair color
                        for j in range(3):
                            if input_label[j] == 1:
                                c[:3] = y[j]
                                break 
                    if i in [0, 2, 3]:   # Gender
                        c[3] = 0 if input_label[3] == 0 else 1
                    if i in [1, 2, 3]:   # Aged
                        c[4] = 0 if input_label[4] == 0 else 1
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        return fixed_c_list
    
    def input_parser(self, input_str):
        words = input_str.split()
        y = [[1, 0, 0],  # black hair
             [0, 1, 0],  # blond hair
             [0, 0, 1]]  # brown hair

        result = [1,0,0,0,1]

        for keys in words:
            if keys == 'black':
       	        result[:3] = y[0]
            elif keys == 'blond':
                result[:3] = y[1]
            elif keys == 'brown':
                result[:3] = y[2]
            if keys in ['man','male']:
                result[3] = 1
            if keys in ['old']:
                result[4] = 0

        return result

    def translate_pic(self, input_str):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.celebA_loader
        else:
            data_loader = self.rafd_loader
        
        #randomly choose a picture in test data
        random.seed()
        for i, (real_x, org_c) in enumerate(data_loader):
            if i == random.randint(0,len(data_loader)): break
        
        real_x = self.to_var(real_x, volatile=True)

        if self.dataset == 'CelebA':
            input_label = self.input_parser(input_str)
            print("input_label",input_label)
            target_c_list = self.make_celeb_labels(org_c, input_label)
        else:
            target_c_list = []
            for j in range(self.c_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                target_c_list.append(self.to_var(target_c, volatile=True))

        # Start translations
        fake_image_list = []
        print(target_c_list[8])
        fake_image_list.append(self.G(real_x, target_c_list[8]))
        #for target_c in target_c_list:
        #    fake_image_list.append(self.G(real_x, target_c))
        fake_images = torch.cat(fake_image_list, dim=3)
        save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
        save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
        print('Translated test images and saved into "{}"..!'.format(save_path))
        img = Image.open(save_path)
        img.show()
