import torch
import os
import argparse
from datasets.crowd import Crowd
from models.bm import BM
from utils.evaluation import eval_game, eval_relative
import numpy as np

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default=r'',
                    help='training data directory')
parser.add_argument('--save-dir',
                    default=r"",
                    help='model directory')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()


def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
            total_params += params
    return total_params

if __name__ == '__main__':
    datasets = Crowd(os.path.join(args.data_dir, 'test'), 256, 8, method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = BM()
    print('Param: {}'.format(count_parameters(model)))
    model.to(device)
    checkpoint = torch.load(args.save_dir, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    i = 1
    epoch_minus = []
    for rgb, t, target, count, name in dataloader:
        rgb = rgb.to(device)
        t = t.to(device)

        assert rgb.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model([rgb, t])
            print(i, name, count.item(), torch.sum(outputs).item())
            epoch_minus.append(torch.sum(outputs).item() - count.item())
            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error
        i += 1

    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '. \
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
