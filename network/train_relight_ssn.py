import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import utils
import time
from tqdm import tqdm
import numpy as np
import os
from os.path import join
import datetime

from ssn.ssn_dataset import SSN_Dataset
# from ssn.ssn_submodule import Contract
from ssn.ssn import Relight_SSN, baseline_2_tbaseline, baseline_2_touchloss
from utils.utils_file import get_cur_time_stamp, create_folder
from utils.net_utils import save_model, get_lr, set_lr
from utils.visdom_utils import setup_visdom, visdom_plot_loss, visdom_log, visdom_show_batch
from utils.tensorboard_utils import *
from torch.utils.tensorboard import SummaryWriter
from params import params as options, parse_params
import matplotlib.pyplot as plt
from evaluation import exp_predict, exp_metric
import pickle

# parse args
params = parse_params()
print("Params: {}".format(params))
exp_name = params.exp_name
# cur_viz = setup_visdom(params.vis_port)
tensorboard_folder = 'tensorboard_log/runs'
os.makedirs(tensorboard_folder, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(join(tensorboard_folder, '{}_{}'.format(exp_name, timestamp)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if params.cpu:
    device = torch.device('cpu')

print("Device: ", device)

""" https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/3
"""
def set_model_optimizer(model, weight_decay):
    optim_params = []
    for key, value in model.named_parameters():
        if not value.requires_grad: continue # frozen weights

        if key[-4:] == 'bias':
            optim_params += [{'params': value,'weight_decay':0.0}]
        else:
            optim_params += [{'params': value,'weight_decay':weight_decay}]

    # import pdb; pdb.set_trace()
    optimizer = optim.Adam(optim_params,
                           lr=params.lr,
                           betas=(params.beta1, 0.999),
                           eps=1e-5)
    # optimizer = optim.SGD(optim_params, lr=params.lr,  momentum=0.9)
    return optimizer

def reconstruct_loss(gt_img, pred_img):
    """ M * (I-I') """
    return torch.norm(gt_img-pred_img, 2)

def get_grid_img(tensor_img, norm=True):
    return utils.make_grid(tensor_img, normalize=norm, nrow=8).detach().cpu().unsqueeze(0)

cur_step = 0
def tensorboard_plot_img(I_t, predicted_img, I_s, L_t, is_training=True, save_batch=False):
    batch_size = min(I_t.shape[0], 4)

    out_channel = predicted_img.shape[1]

    # import pdb; pdb.set_trace()
    vis_predicted_img = get_grid_img(predicted_img[:batch_size, 0:1, :, :], True)
    vis_predicted_img_gt = get_grid_img(I_t[:batch_size, :1, :, :], True)

    if out_channel == 2:
        vis_touch_img = get_grid_img(predicted_img[:batch_size, -1:, :, :], True)
        vis_touch_img_gt = get_grid_img(I_t[:batch_size, -1:, :, :], True)

    if save_batch:
        vis_predicted_img_np = np.clip(vis_predicted_img[0].detach().cpu().numpy().transpose((1, 2, 0)), 0.0, 1.0)
        vis_predicted_img_gt_np = np.clip(vis_predicted_img_gt[0].detach().cpu().numpy().transpose((1, 2, 0)), 0.0, 1.0)
        saving_folder = 'training_result'
        pred_fname, gt_fname = os.path.join(saving_folder,
                                            'predict_{}.png'.format(datetime.datetime.now())), os.path.join(
            saving_folder, 'gt_{}.png'.format(datetime.datetime.now()))
        plt.imsave(pred_fname, vis_predicted_img_np, cmap='gray')
        plt.imsave(gt_fname, vis_predicted_img_gt_np, cmap='gray')

    vis_shadow_img = torch.cat((vis_predicted_img_gt, vis_predicted_img))
    if out_channel == 2:
        vis_touch_img = torch.cat((vis_touch_img_gt, vis_touch_img))

    if is_training:
        win_prefix = 'train'
    else:
        win_prefix = 'valid'

    channel = I_s.shape[1]
    for i in range(channel):
        cur_channel = I_s[:batch_size, i:i + 1, :, :]
        cur_channel = get_grid_img(cur_channel, True)
        tensorboard_show_batch(cur_channel, writer, win_name="{} {}".format(win_prefix, i), nrow=4, normalize=False, step=cur_step)

    tensorboard_show_batch(vis_shadow_img, writer, win_name="{} shadow gt vs. inference".format(exp_name,win_prefix), nrow=1,
                      normalize=False, step=cur_step)
    tensorboard_show_batch(get_grid_img(L_t[:batch_size]), writer, win_name='{} light'.format(win_prefix), normalize=True, step=cur_step)
    if out_channel == 2:
        tensorboard_show_batch(vis_touch_img, writer, win_name="{} touch gt vs. inference".format(win_prefix), nrow=1,
                          normalize=False, step=cur_step)

ao_loss = []
def training_iteration(model, train_dataloder, optimizer, train_loss, epoch_num):
    # training
    cur_epoch_loss = 0.0
    model.train()

    with tqdm(total=len(train_dataloder) * params.timers) as t:
        t.set_description("Ep. {}".format(epoch_num))
        for j in range(params.timers):
            for i, gt_data in enumerate(train_dataloder):
                cur_step = i
                inputs, light, shadow = gt_data[0], gt_data[1], gt_data[2]
                I_s, L_t, I_t = inputs.to(device), light.to(device), shadow.to(device)
                optimizer.zero_grad()

                mask, touch = I_s[:,:1,:,:], I_s[:,-1:,:,:]

                # predict
                if params.input_channel == 1:
                    I_s = mask

                predicted_img, predicted_src_light = model(I_s, L_t)

                # compute loss
                if params.pred_touch:
                    I_t = touch

                if params.touch_loss:
                    I_t = torch.cat((I_t, touch * 30.0), axis=1)

                loss = reconstruct_loss(I_t, predicted_img)

                if params.touch_loss:
                    touch_loss = reconstruct_loss(I_t[:,-1:,:,:], predicted_img[:,-1:,:,:])/np.sqrt(params.batch_size)
                    ao_loss.append(touch_loss.item())
                    tensorboard_plot_loss("train_ao_loss", ao_loss, writer)

                loss.backward()
                optimizer.step()

                cur_epoch_loss += loss.item()

                # visualize results
                if i % 10 == 0:
                    tensorboard_plot_img(I_t, predicted_img, inputs, L_t, save_batch=params.save)

                # keep tracking
                train_loss.append(loss.item()/np.sqrt(params.batch_size))
                tensorboard_plot_loss("train_total_loss", train_loss, writer)

                t.update()

    # Finish one epoch
    cur_epoch_loss /= (params.timers * len(train_dataloder) * np.sqrt(params.batch_size))
    return cur_epoch_loss

def validation_iteration(model, valid_dataloader, valid_loss, epoch_num):
    cur_epoch_loss = 0.0
    model.eval()

    cur_timer = params.timers
    with torch.no_grad():
        with tqdm(total=len(valid_dataloader) * params.timers) as t:
            t.set_description("(Validation)Ep. {} ".format(epoch_num))
            for j in range(cur_timer):
                for i, gt_data in enumerate(valid_dataloader):
                    inputs, light, shadow = gt_data[0], gt_data[1], gt_data[2]
                    I_s, L_t, I_t = inputs.to(device), light.to(device), shadow.to(device)
                    mask, touch = I_s[:,:1,:,:], I_s[:,-1:,:,:]

                    # predict
                    if params.input_channel==1:
                        I_s = mask

                    predicted_img, predicted_src_light = model(I_s, L_t)

                    # compute loss
                    if params.pred_touch:
                        I_t = touch

                    # predict transfer
                    predicted_img, predicted_src_light = model(I_s, L_t)

                    if params.touch_loss:
                        predicted_img = predicted_img[:,:1,:,:]

                    # compute loss
                    loss = reconstruct_loss(I_t, predicted_img)

                    cur_epoch_loss += loss.item()

                    cur_step = i
                    # visualize results
                    if i % 10 == 0:
                        tensorboard_plot_img(I_t, predicted_img, inputs, L_t, False)

                    # keep tracking
                    valid_loss.append(loss.item()/np.sqrt(params.batch_size))

                    tensorboard_plot_loss("valid_total_loss", valid_loss, writer)
                    t.update()

    # Finish one epoch
    # import pdb; pdb.set_trace()
    cur_epoch_loss /= (np.sqrt(params.batch_size) * len(valid_dataloader) * cur_timer)

    return cur_epoch_loss

def train(params):
    # history logs
    best_valid_loss = float('inf')
    log_info = ""
    hist_train_loss, hist_valid_loss, hist_lr = [], [], []

    # dataset
    # ds_csv = "/home/ysheng/Dataset/new_dataset/meta_data.csv"
    # ds_folder = './dataset/new_dataset'
    ds_folder = params.ds_folder
    train_set = SSN_Dataset(ds_folder, True)
    train_dataloder = DataLoader(train_set, batch_size= min(len(train_set), params.batch_size), shuffle=True, num_workers=params.workers, drop_last=True)
    valid_set = SSN_Dataset(ds_folder, False)
    valid_dataloader = DataLoader(valid_set, batch_size= min(len(valid_set), params.batch_size), shuffle=False, num_workers=params.workers, drop_last=True)

    best_weight = ''
    # model & optimizer & scheduler & loss function
    if not params.from_baseline:
        input_channel = params.input_channel
        model = Relight_SSN(input_channel, 1)    # input is mask + human
        model.to(device)
    else:
        model = Relight_SSN(1, 1)
        model.to(device)
        baseline_checkpoint = torch.load("weights/human_baseline.pt", map_location=device)
        model.load_state_dict(baseline_checkpoint['model_state_dict'])

    if params.tbaseline:
        params.input_channel = 2
        model = baseline_2_tbaseline(model)
        model.to(device)

    if params.touch_loss:
        params.input_channel = 1
        model = baseline_2_touchloss(model)
        model.to(device)

    # resume from last saved points
    if params.resume:
        best_weight = os.path.join("weights", params.weight_file)
        checkpoint = torch.load(best_weight, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint['best_loss']
        hist_train_loss = checkpoint['hist_train_loss']
        hist_valid_loss = checkpoint['hist_valid_loss']
        if 'hist_lr' in checkpoint.keys():
            hist_lr = checkpoint['hist_lr']
        print("resuming from: {}".format(best_weight))
        del checkpoint

        # tensorboard writer update history
        for i in range(0, len(hist_train_loss)):
            tensorboard_plot_loss("history train loss", hist_train_loss[:i+1], writer)

        for i in range(0, len(hist_valid_loss)):
            tensorboard_plot_loss("history valid loss", hist_valid_loss[:i+1], writer)

    if params.relearn:
        best_valid_loss = float('inf')

    print(torch.cuda.device_count())
    # test multiple GPUs
    if torch.cuda.device_count() > 1 and params.multi_gpu:
        print("Let's use ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    optimizer = set_model_optimizer(model, params.weight_decay)
    set_lr(optimizer, params.lr)
    print("Current LR: {}".format(get_lr(optimizer)))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params.patience)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)

    # training states
    train_loss, valid_loss = [], []

    # training iterations
    for epoch in range(params.epochs):
        # training
        cur_train_loss = training_iteration(model, train_dataloder, optimizer, train_loss, epoch)

#         # validation
        cur_valid_loss = validation_iteration(model, valid_dataloader, valid_loss, epoch)

        if params.use_schedule:
            scheduler.step()

        log_info += "Current epoch: {} Learning Rate: {}  <br>".format(epoch, get_lr(optimizer))
        tensorboard_log(log_info, writer, step=epoch)

        hist_train_loss.append(cur_train_loss)
        hist_valid_loss.append(cur_valid_loss)

        tensorboard_plot_loss("history train loss", hist_train_loss, writer)
        tensorboard_plot_loss("history valid loss", hist_valid_loss, writer)

        log_info += "Epoch: {} training loss: {}, valid loss: {}  <br>".format(epoch, cur_train_loss, cur_valid_loss)
        # save results
        if best_valid_loss > cur_valid_loss:
            log_info += "<br> ---------- Exp: {} Find better loss: {} at {} --------  <br>".format(exp_name, cur_valid_loss, datetime.datetime.now())
            tensorboard_log(log_info, writer, step=epoch)

            best_valid_loss = cur_valid_loss

            outfname = '{}_{}.pt'.format(exp_name, get_cur_time_stamp())
            best_weight = save_model("weights", model, optimizer, epoch, best_valid_loss, outfname, hist_train_loss, hist_valid_loss, hist_lr, params)

        outfname = '{}.pt'.format(exp_name)
        save_model("weights", model, optimizer, epoch, best_valid_loss, outfname, hist_train_loss, hist_valid_loss, hist_lr, params)

        # saving loss to local directory
        plt.figure()
        plt.plot(hist_train_loss, label='train loss')
        plt.plot(hist_train_loss, label='valid loss')
        plt.legend()
        plt.savefig('{}_loss_plot.png'.format(params.exp_name))
        plt.close()

        # termination
        if get_lr(optimizer) < 1e-7:
            break

    print("Training finished")
    return best_weight

if __name__ == "__main__":
    best_weight, best_valid_loss = train(params)
