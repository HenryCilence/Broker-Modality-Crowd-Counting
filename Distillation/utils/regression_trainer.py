from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import torch
import time
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
from tqdm import tqdm
import math


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.crowd import Crowd
from models.unet_cross_attention import U_Net


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    rgb = torch.stack(transposed_batch[0], 0)
    t = torch.stack(transposed_batch[1], 0)
    rgbt = torch.stack(transposed_batch[2], 0)
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return rgb, t, rgbt, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size, x) for x in ['train', 'val']}
        self.dataloaders = DataLoader(self.datasets["train"],
                                      collate_fn=train_collate,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers * self.device_count,
                                      pin_memory=True)
        self.test_dataloader = DataLoader(self.datasets["val"], 1, shuffle=False,
                                                           num_workers=8, pin_memory=False)
        self.model = U_Net()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.criterion = torch.nn.MSELoss()
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mse = np.inf
        self.best_epoch = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.test_epoch()

    def train_epoch(self):
        epoch_start = time.time()
        epoch_loss = AverageMeter()

        self.model.train()  # Set model to training mode
        dataloader = tqdm(self.dataloaders, desc="Training", leave=False, dynamic_ncols=True)

        # Iterate over data.
        for step, (rgb, t, fusion, st_sizes) in enumerate(dataloader):
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            fusion = fusion.to(self.device)          

            with torch.set_grad_enabled(True):
                outputs = self.model(rgb, t)
                loss = self.criterion(fusion, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = rgb.size(0)
                epoch_loss.update(loss.item(), N)
        dataloader.close()

        logging.info('Epoch {} Train, Loss: {}, Cost {:.2f} sec'
                     .format(self.epoch, epoch_loss.get_avg() * 255, time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def test_epoch(self):
        epoch_start = time.time()
        epoch_test_loss = AverageMeter()
        self.model.eval()  # Set model to evaluate mode
        
        dataloader = tqdm(self.test_dataloader, desc="Testing", leave=False, dynamic_ncols=True)

        for rgb, t, fusion, name in dataloader:
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            fusion = fusion.to(self.device)

            assert rgb.size(0) == 1, 'the batch size should equal to 1 in validation mode'

            with torch.set_grad_enabled(False):
                outputs = self.model(rgb, t)
                test_loss = self.criterion(fusion, outputs)
                epoch_test_loss.update(test_loss.item(), 1)
        dataloader.close()

        N = len(self.test_dataloader)
        model_state_dic = self.model.state_dict()

        log_str = 'Test {}, Loss {test_avg_loss:}, Time cost {time_cost:.2f} sec'. \
            format(N, test_avg_loss=epoch_test_loss.get_avg() * 255, time_cost=time.time() - epoch_start)
        logging.info(log_str)

        if epoch_test_loss.get_avg() < self.best_mse:
            self.best_mse = epoch_test_loss.get_avg()
            self.best_epoch = self.epoch
            logging.info(
                "***** Save Best Loss {test_avg_loss:} Model Epoch {e}".format(
                    test_avg_loss=epoch_test_loss.get_avg() * 255,
                    e=self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        else:
            logging.info('Best Loss {test_avg_loss:} Epoch {e}'.format(test_avg_loss=self.best_mse * 255,
                                                                       e=self.best_epoch))
