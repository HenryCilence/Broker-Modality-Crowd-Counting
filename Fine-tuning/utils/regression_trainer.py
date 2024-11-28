from tqdm import tqdm
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
import math
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluation import eval_game, eval_relative
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from models.bm import BM


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    rgb = torch.stack(transposed_batch[0], 0)
    t = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[3]
    st_sizes = torch.FloatTensor(transposed_batch[4])
    return rgb, t, points, targets, st_sizes


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

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio, x) for x in ['train', 'val', 'test']}
        self.dataloaders = DataLoader(self.datasets['train'],
                                      collate_fn=train_collate,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers * self.device_count,
                                      pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.datasets['val'], 1, shuffle=False,
                                                           num_workers=8, pin_memory=False)
        self.test_dataloader = torch.utils.data.DataLoader(self.datasets['test'], 1, shuffle=False,
                                                           num_workers=8, pin_memory=False)
        self.model = BM()
        self.model.to(self.device)
        if args.pretrained_model != "":
            print(self.model.load_state_dict(torch.load(args.pretrained_model), strict=False))
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

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_game0 = np.inf
        self.best_game3 = np.inf
        self.best_epoch = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                game0_is_best, game3_is_best = self.val_epoch()
            if epoch >= args.val_start and (game0_is_best or game3_is_best):
                self.test_epoch()

    def train_epoch(self):
        epoch_start = time.time()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()  # Set model to training mode

        # Use tqdm to create a progress bar
        dataloader = tqdm(self.dataloaders, desc="Training", leave=False, dynamic_ncols=True)

        # Iterate over data.
        for step, (rgb, t, points, targets, st_sizes) in enumerate(dataloader):
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs = self.model([rgb, t])
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = rgb.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        dataloader.close()

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models   

    def val_epoch(self):
        epoch_start = time.time()
        args = self.args
        self.model.eval()  # Set model to evaluate mode

        # Iterate over data.
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0
        dataloader = tqdm(self.val_dataloader, desc="Validating", leave=False, dynamic_ncols=True)

        for rgb, t, target, count, name in dataloader:
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            assert rgb.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model([rgb, t])
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
        dataloader.close()

        N = len(self.val_dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        logging.info('Epoch {} Val{}, '
                    'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} Re {relative:.4f}, Time cost {time_cost:.1f}s'
                    .format(self.epoch, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error, time_cost=time.time() - epoch_start
                            )
                    )

        model_state_dic = self.model.state_dict()

        game0_is_best = game[0] < self.best_game0
        game3_is_best = game[3] < self.best_game3

        if game[0] < self.best_game0 or game[3] < self.best_game3:
            self.best_game3 = min(game[3], self.best_game3)
            self.best_game0 = min(game[0], self.best_game0)
            logging.info("*** Best Val GAME0 {:.3f} GAME3 {:.3f} model epoch {}".format(self.best_game0,
                                                                                    self.best_game3,
                                                                                    self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        return game0_is_best, game3_is_best

    def test_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0

        # Use tqdm to create a progress bar
        dataloader = tqdm(self.test_dataloader, desc="Testing", leave=False, dynamic_ncols=True)

        for rgb, t, target, count, name in dataloader:
            rgb = rgb.to(self.device)
            t = t.to(self.device)

            assert rgb.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model([rgb, t])
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
        dataloader.close()

        N = len(self.test_dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N
        log_str = 'Test {}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f} Re {relative:.4f}, Time cost {time_cost:.1f}s'. \
            format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],
                   relative=total_relative_error, time_cost=time.time() - epoch_start)
        logging.info(log_str)
