import os
from os.path import join
import logging
from collections import OrderedDict
from glob import glob

from tqdm import tqdm
from time import time
import torch
import torch.nn as nn

import models
import datasets

from utils import utils
from utils import vis_writer

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.setup()


    def setup(self):
        """ Setup Training settings:
            1. Dataloader
            2. Model
            3. Hyper-Params
        """
        opt             = self.opt
        exp_name        = opt['exp_name']
        self.vis_iter   = opt['hyper_params']['vis_iter']
        self.save_iter  = opt['hyper_params']['save_iter']
        self.log_folder = join(opt['hyper_params']['default_folder'], exp_name)
        self.cur_epoch  = 0

        utils.logging_init(opt['exp_name'])
        os.makedirs(self.log_folder, exist_ok=True)


        if not torch.cuda.is_available():
            logging.warn('Not GPU found! Use cpu')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        """ Prepare Dataloader """
        dataloaders           = datasets.create_dataset(opt)
        self.train_dataloader = dataloaders['train']
        self.eval_dataloader  = dataloaders['eval']

        """ Prepare Model """
        self.model = models.create_model(opt)

        """ Prepare Hyper Params """
        self.hist_loss = {'epoch_train_loss': [],
                          'epoch_eval_loss': [],
                          'iter_train_loss': [],
                          'iter_eval_loss': []}

        """ Prepare Visualizer's writer """
        self.exp_logger = vis_writer.vis_writer(opt)

        """ Logging All Params """
        self.log_all_params()

        """ Setup Training """
        self.setup_training()


    def setup_training(self):
        """ Setup states before training
              - Resume?
                - resume model, optimzer's states
                - resume history loss
              - Which GPU?
        """
        opt         = self.opt
        model       = self.model
        resume      = opt['hyper_params']['resume']
        weight_file = opt['hyper_params']['weight_file']
        devices     = opt['hyper_params']['gpus']

        models     = model.get_models()
        if torch.cuda.is_available():
            for k, m in models.items():
                if len(devices) > 1: # mutliple GPU
                    logging.info('Use GPU: {}'.format(','.join([str(d) for d in devices])))
                    models[k] = nn.DataParallel(m, device_ids=devices)
                models[k].to(self.device)

        if resume:
            self.resume(weight_file)

        model.set_models(models)
        self.model = model


    def fit(self, dataloader, is_training):
        """ Fittin the current dataset
        """
        vis_iter  = self.vis_iter
        save_iter = self.save_iter
        model     = self.model
        epoch     = self.cur_epoch
        models    = model.get_models()

        if is_training:
            torch.set_grad_enabled(True)
            desc = 'Training:'
        else:
            torch.set_grad_enabled(False)
            desc = 'Eval'

        for k, m in models.items():
            if is_training:
                m.train()
            else:
                m.eval()

        # begin fitting
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=desc)):
            x = {k: v.to(self.device) for k, v in data['x'].items()}
            y = data['y'].to(self.device)

            train_x = model.setup_input(x)
            loss    = model.supervise(train_x, y, is_training)

            # record loss
            epoch_loss += loss
            self.add_hist_iter_loss(loss, is_training)

            # record model's losses
            logs = model.get_logs()
            self.logging(logs, is_training)

            # visualization
            if i % vis_iter == 0:
                vis_imgs = model.get_visualize()
                self.visualize(vis_imgs, is_training)

            # we save some visualization results into a html file
            if i % save_iter == 0:
                vis_imgs = model.get_visualize()
                self.save_visualize(vis_imgs, epoch, i, is_training)

        # plot the epoch loss
        epoch_loss = epoch_loss / len(dataloader)
        self.add_hist_epoch_loss(epoch_loss, is_training)
        return epoch_loss


    def train(self):
        """  Training the model
                For i in epochs:
                data = ?
                pred = model.forward(x)
                loss = model.compute_loss(y, pred)
                model.optimize(loss)
        """
        exp_name         = self.opt['exp_name']
        hyper_params     = self.opt['hyper_params']
        train_dataloader = self.train_dataloader
        eval_dataloader  = self.eval_dataloader
        start_epoch      = self.cur_epoch

        epochs     = hyper_params['epochs']
        save_epoch = hyper_params['save_epoch']
        desc       = 'Exp. {}'.format(exp_name)

        pbar = tqdm(range(start_epoch, epochs), desc=desc)
        for epoch in pbar:
            train_epoch_loss = self.fit(train_dataloader, is_training=True)
            eval_epoch_loss  = self.fit(eval_dataloader, is_training=False)

            # save model
            if epoch % save_epoch == 0:
                self.save(epoch)

            # plotting epoch loss
            desc = 'Exp. {}, Train loss: {}, Eval loss: {}'.format(exp_name,
                                                                   train_epoch_loss,
                                                                   eval_epoch_loss)

            pbar.set_description(desc)
            # plotting epoch loss together
            self.exp_logger.plot_losses({'train': train_epoch_loss, 'eval': eval_epoch_loss}, 'All')
            self.cur_epoch = epoch + 1


    def log_all_params(self):
        opt      = self.opt

        logging.info('')
        logging.info('#' * 60)
        logging.info('Training params:')
        logging.info('Model:')
        logging.info('{}'.format(str(opt['model'])))
        logging.info('-' * 60)

        logging.info('Dataset:')
        logging.info('{}'.format(str(opt['dataset'])))
        logging.info('-' * 60)

        logging.info('Hyper Params')
        logging.info('{}'.format(str(opt['hyper_params'])))
        logging.info('-' * 60)

        logging.info('Model size')
        logging.info('{} MB'.format(self.get_model_size()))
        logging.info('-' * 60)
        logging.info('#' * 60)
        logging.info('')


    def add_hist_epoch_loss(self, loss, is_training):
        if is_training:
            key = 'epoch_train_loss'
        else:
            key = 'epoch_eval_loss'

        self.hist_loss[key].append(loss)
        self.exp_logger.plot_loss(loss, key)


    def add_hist_iter_loss(self, loss, is_training):
        if is_training:
            key = 'iter_train_loss'
        else:
            key = 'iter_eval_loss'

        self.hist_loss[key].append(loss)
        self.exp_logger.plot_loss(loss, key)


    def save(self, epoch):
        """ Note, we only save:
               - Models
               - Optimizers
               - history loss
        """
        opt        = self.opt
        gpus       = opt['hyper_params']['gpus']
        model      = self.model
        hist_loss  = self.hist_loss
        log_folder = self.log_folder

        ofile_name = join(log_folder, '{:010d}.pt'.format(epoch))

        tmp_model      = model.get_models()
        tmp_optimizers = model.get_optimizers()

        if len(gpus) > 1:
            for k, v in tmp_model.items():
                tmp_model[k] = v.module

        save_dict = {k:v.state_dict() for k, v in tmp_model.items()}
        for k, opt in tmp_optimizers.items():
            save_dict[k] = opt.state_dict()

        save_dict['hist_loss'] = hist_loss
        save_dict['cur_epoch'] = epoch
        torch.save(save_dict, ofile_name)


    def resume(self, weight_file):
        """ Resume from file
                - Models
                - Optimizers
                - History Loss
        """
        model      = self.model
        models     = self.model.get_models()
        optimizers = self.model.get_optimizers()
        hist_loss  = self.hist_loss
        cur_epoch  = self.cur_epoch
        device     = self.device

        log_folder = self.log_folder

        if weight_file  == 'latest':
            files  = glob(join(log_folder, '*.pt'))

            if len(files) == 0:
                err = 'There is no *.pt file in {}'.format(log_folder)
                logging.error(err)
                raise ValueError(err)

            files.sort()

            weight_file = files[-1]
            logging.info('Resume from file {}'.format(weight_file))
        else:
            weight_file = join(log_folder, weight_file)

            if not os.path.exists(weight_file):
                err = 'There is no {} file'.format(weight_file)
                logging.error(err)
                raise ValueError(err)

        checkpoint = torch.load(weight_file, map_location=device)

        for k, m in models.items():
            models[k].load_state_dict(checkpoint[k])

        for k, o in optimizers.items():
            optimizers[k].load_state_dict(checkpoint[k])

        hist_loss  = checkpoint['hist_loss']
        for k, v in hist_loss.items():
            for l in v:
                self.exp_logger.plot_loss(l, k)

        cur_epoch  = checkpoint['cur_epoch']
        model.set_models(models)
        model.set_optimizers(optimizers)

        self.model     = model
        self.hist_loss = hist_loss
        self.cur_epoch = cur_epoch


    def visualize(self, vis_imgs: OrderedDict, is_training:bool):
        prefix = 'Train'
        if not is_training:
            prefix = 'Valid'

        counter = 0
        for k, v in vis_imgs.items():
            name = '{}/{:02d}_{}'.format(prefix, counter, k)
            self.exp_logger.plot_img(v, name)

            counter += 1


    def logging(self, logs, is_training):
        if logs is None:
            return

        prefix = 'Train'
        if not is_training:
            prefix = 'Valid'

        for k, v in logs.items():
            name = '{}/{}'.format(prefix, k)
            self.exp_logger.plot_loss(v, name)
            logging.info('{}: {}'.format(name, v))



    def save_visualize(self, vis_imgs: OrderedDict, epoch, iteration, is_training):
        if is_training:
            label = '{}_{:010d}_{:010d}'.format('Train', epoch, iteration)
        else:
            label = '{}_{:010d}_{:010d}'.format('Eval', epoch, iteration)
        self.exp_logger.save_visualize(vis_imgs, label)


    def get_model_size(self):
        model = self.model

        total_size = 0
        models = model.get_models()
        for k, m in models.items():
            total_size += utils.get_model_size(m)

        # return model size in mb
        return total_size/(1024 ** 2)


if __name__ == '__main__':
    opt = utils.parse_configs()

    trainer = Trainer(opt)
    trainer.train()
