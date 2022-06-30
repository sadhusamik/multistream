#!/usr/bin/env python3

import os
import logging
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
from src.nnet.models.VAE import nnetVAE, VAE_LSTM, FF_VAE, get_detailed_loss, save_fig, FF_VAE_Rotation
from src.nnet.dataprep.datasets import nnetDatasetSeq, nnetDatasetSeqAE
import pickle as pkl
from dotmap import DotMap
from pathlib import Path

import subprocess


def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err


def vae_loss(x, ae_out, latent_out, out_dist='gauss'):
    if out_dist == 'gauss':
        log_lhood = torch.mean(-0.5 * torch.pow((x - ae_out), 2) - 0.5 * np.log(2 * np.pi * 1))
    elif out_dist == 'laplace':
        log_lhood = torch.mean(-torch.abs(x - ae_out) - np.log(2))
    else:
        logging.error("Output distribution of VAE can be 'gauss' or 'laplace'")
        sys.exit(1)

    kl_loss = 0.5 * torch.mean(
        1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1])
    return log_lhood, kl_loss


def ae_loss(x, ae_out, out_dist='gauss'):
    if out_dist == 'gauss':
        loss = torch.mean(torch.pow((x - ae_out), 2))
    elif out_dist == 'laplace':
        loss = torch.mean(torch.abs(x - ae_out))
    else:
        logging.error("Output distribution of VAE can be 'gauss' or 'laplace'")
        sys.exit(1)

    return loss


def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a VAE")

    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    parser.add_argument("--encoder_num_layers", default=3, type=int, help="Number of encoder layers")
    parser.add_argument("--decoder_num_layers", default=1, type=int, help="Number of decoder layers")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Number of hidden nodes")
    parser.add_argument("--bn_dim", default=30, type=int, help="Bottle neck dim")

    # Training configuration
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
    parser.add_argument("--batch_size", default=64, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--train_set", default="train_si284", help="Name of the training datatset")
    parser.add_argument("--dev_set", default="test_dev93", help="Name of development dataset")
    parser.add_argument("--clip_thresh", type=float, default=1, help="Gradient clipping threshold")
    parser.add_argument("--lrr", type=float, default=0.9, help="Learning rate reduction rate")
    parser.add_argument("--lr_tol", type=float, default=0.01,
                        help="Percentage of tolerance to leave on dev error for lr scheduling")
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 Regularization weight")
    parser.add_argument("--enc_var_init", type=float, default=None,
                        help="Set to a value to initialize the log_var of encoder")
    parser.add_argument("--unit_decoder_var", action="store_true",
                        help="Set to a value to use unit variance for output")
    parser.add_argument("--shared_decoder_var", action="store_true",
                        help="Set to a value to use shared decoder variance")

    # Misc configurations
    parser.add_argument("--resume_checkpoint", default=None, help="Set a checkpoint to load model from")
    parser.add_argument("--use_transformer", action="store_true", help="Set to use transformer layers instead of RNN")
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
    parser.add_argument("--model_save_interval", type=int, default=10,
                        help="Number of epochs to skip before every model save")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--load_data_workers", default=5, type=int, help="Number of parallel data loaders")
    parser.add_argument("--experiment_name", default="exp_run", type=str, help="Name of this experiment")
    parser.add_argument("--out_dist", default="gauss", help="Output distribution of VAE, 'gauss' or 'laplace'")
    parser.add_argument("--only_AE", action="store_true", help="Will train only an Autoencoder and not VAE")

    return parser.parse_args()


def run(config):
    model_dir = os.path.join(config.store_path, config.experiment_name + '.dir')
    os.makedirs(config.store_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(model_dir, config.experiment_name),
        filemode='w')

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Load feature configuration
    egs_config = pkl.load(open(os.path.join(config.egs_dir, config.train_set, 'egs.config'), 'rb'))
    context = egs_config['concat_feats']

    num_frames = 0
    if context is not None:
        context = context.split(',')
        num_frames += int(context[0]) + int(context[1]) + 1
    else:
        num_frames += 1

    # Prepare model config file
    conf = {}
    conf['input_size'] = config.feature_dim * num_frames
    conf['num_layers_enc'] = config.encoder_num_layers
    conf['num_layers_dec'] = config.decoder_num_layers
    conf['hidden_size'] = config.hidden_dim
    conf['bn_size'] = config.bn_dim
    conf['dropout'] = 0  # We dont use any dropout
    conf['out_dist'] = config.out_dist
    conf['only_AE'] = config.only_AE
    conf['unit_decoder_var'] = config.unit_decoder_var
    conf['shared_decoder_var'] = config.shared_decoder_var
    conf['use_transformer'] = config.use_transformer

    logging.info('Model Parameters: ')
    logging.info('Encoder Number of Layers: %d' % (config.encoder_num_layers))
    logging.info('Decoder Number of Layers: %d' % (config.decoder_num_layers))
    logging.info('Hidden Dimension: %d' % (config.hidden_dim))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Bottleneck dimension: %d' % (config.bn_dim))
    logging.info('Number of Frames: %d' % (num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
    logging.info('Learning rate reduction rate: %f ' % (config.lrr))
    logging.info('Weight decay: %f ' % (config.weight_decay))
    logging.info('Output distribution: %s ' % (config.out_dist))


    if config.only_AE:
        logging.info('Training only an Autoencoder')
    if config.use_transformer:
        logging.info('Training with Transformer layers instead of RNN')
    if config.unit_decoder_var:
        logging.info('Using unit variance for decoder output')
    else:
        logging.info('Learning variance of decoder output')
    sys.stdout.flush()

    # Define Model
    model = VAE_LSTM(conf, use_gpu=config.use_gpu)

    if config.resume_checkpoint:
        ckpt = torch.load(config.resume_checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch_start = ckpt['epoch'] - 1
    else:
        epoch_start = 0
        if config.enc_var_init:
            logging.info('Initializing the log variance of encoder with: %f' % (config.enc_var_init))
            model.init_enc_var(config.enc_var_init)
            conf['enc_var_init'] = config.enc_var_init

    if config.use_gpu:
        # Set environment variable for GPU ID
        id = get_device_id()
        os.environ["CUDA_VISIBLE_DEVICES"] = id

        model = model.cuda()

    lr = config.learning_rate

    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adamnomom':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                               betas=(0, 0.9))
    elif config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError("Learning method not supported for the task")

    if config.resume_checkpoint:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

    if not config.resume_checkpoint:
        model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
        torch.save({
            'epoch': 1,
            'conf': conf,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

    if config.only_AE:
        ep_ae_rec_tr = []
        ep_ae_rec_dev = []
    else:
        ep_vae_rec_tr = []
        ep_vae_kl_tr = []
        ep_vae_rec_dev = []
        ep_vae_kl_dev = []
        ep_enc_var_tr = []
        ep_dec_var_tr = []
        ep_enc_var_dev = []
        ep_dec_var_dev = []

    # Load Datasets
    dataset_train = nnetDatasetSeqAE(os.path.join(config.egs_dir, config.train_set))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                                    num_workers=config.load_data_workers)

    dataset_dev = nnetDatasetSeqAE(os.path.join(config.egs_dir, config.dev_set))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True,
                                                  num_workers=config.load_data_workers)

    err_p = np.inf
    best_model_state = None

    for epoch_i in range(epoch_start, config.epochs):

        ####################
        ##### Training #####
        ####################

        model.train()
        if config.only_AE:
            train_ae_losses = []
        else:
            train_vae_rec_losses = []
            train_vae_kl_losses = []
            train_enc_var = []
            train_dec_var = []

        # Main training loop

        for batch_x, batch_l in data_loader_train:
            _, indices = torch.sort(batch_l, descending=True)
            if config.use_gpu:
                batch_x = batch_x[indices].cuda()
                batch_l = batch_l[indices]
            else:
                batch_x = batch_x[indices]
                batch_l = batch_l[indices]

            optimizer.zero_grad()

            # Main forward pass
            ae_out, latent_out = model(batch_x, batch_l)

            if config.only_AE:
                loss = model.ae_loss_train(batch_x, batch_l, ae_out, out_dist=config.out_dist)
                train_ae_losses.append(loss.item())
                (-loss).backward()
            else:
                loss = model.vae_loss_train(batch_x, batch_l, ae_out, latent_out)
                var_enc, var_dec = model.compute_encoder_and_decoder_vars(batch_x, batch_l, ae_out, latent_out)
                train_enc_var.append(var_enc)
                train_dec_var.append(var_dec)
                train_vae_rec_losses.append(loss[0].item())
                train_vae_kl_losses.append(loss[1].item())
                (-loss[0] + loss[1]).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

        if config.only_AE:
            ep_ae_rec_tr.append(np.mean(train_ae_losses))
        else:
            ep_vae_rec_tr.append(np.mean(train_vae_rec_losses))
            ep_vae_kl_tr.append(np.mean(train_vae_kl_losses))
            ep_enc_var_tr.append(np.mean(train_enc_var))
            ep_dec_var_tr.append(np.mean(train_dec_var))

        ######################
        ##### Validation #####
        ######################

        model.eval()

        with torch.set_grad_enabled(False):

            if config.only_AE:
                val_ae_losses = []
            else:
                val_vae_rec_losses = []
                val_vae_kl_losses = []
                val_enc_var = []
                val_dec_var = []

            for batch_x, batch_l in data_loader_dev:
                _, indices = torch.sort(batch_l, descending=True)
                if config.use_gpu:
                    batch_x = batch_x[indices].cuda()
                    batch_l = batch_l[indices]
                else:
                    batch_x = batch_x[indices]
                    batch_l = batch_l[indices]

                # Main forward pass
                ae_out, latent_out = model(batch_x, batch_l)

                # Convert all the weird tensors to frame-wise form

                if config.only_AE:
                    loss = model.ae_loss_train(batch_x, batch_l, ae_out, out_dist=config.out_dist)
                    val_ae_losses.append(loss.item())
                else:
                    loss = model.vae_loss_train(batch_x, batch_l, ae_out, latent_out)
                    var_enc, var_dec = model.compute_encoder_and_decoder_vars(batch_x, batch_l, ae_out, latent_out)
                    val_vae_rec_losses.append(loss[0].item())
                    val_vae_kl_losses.append(loss[1].item())
                    val_enc_var.append(var_enc)
                    val_dec_var.append(var_dec)
                    ep_enc_var_dev.append(np.mean(val_enc_var))
                    ep_dec_var_dev.append(np.mean(val_dec_var))

            if config.only_AE:
                ep_ae_rec_dev.append(np.mean(val_ae_losses))
            else:
                ep_vae_rec_dev.append(np.mean(val_vae_rec_losses))
                ep_vae_kl_dev.append(np.mean(val_vae_kl_losses))

        # Manage learning rate
        if config.only_AE:
            if epoch_i == 0:
                err_p = -np.mean(val_ae_losses)
                best_model_state = model.state_dict()
            else:
                if -np.mean(val_ae_losses) > (100 - config.lr_tol) * err_p / 100:
                    logging.info(
                        "Val loss went up, Changing learning rate from {:.6f} to {:.6f}".format(lr, config.lrr * lr))
                    lr = config.lrr * lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    model.load_state_dict(best_model_state)
                else:
                    err_p = -np.mean(val_ae_losses)
                    best_model_state = model.state_dict()

            print_log = "Epoch: {:d} ((lr={:.6f})) Tr AE llhood: {:.3f} :: Val AE llhood: {:.3f}".format(
                epoch_i + 1, lr, ep_ae_rec_tr[-1], ep_ae_rec_dev[-1])

            logging.info(print_log)

            if (epoch_i + 1) % config.model_save_interval == 0:
                model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
                torch.save({
                    'epoch': epoch_i + 1,
                    'conf': conf,
                    'feature_dim': config.feature_dim,
                    'num_frames': num_frames,
                    'encoder_num_layers': config.encoder_num_layers,
                    'decoder_num_layers': config.decoder_num_layers,
                    'hidden_dim': config.hidden_dim,
                    'bn_dim': config.bn_dim,
                    'ep_ae_rec_tr': ep_ae_rec_tr,
                    'ep_ae_rec_dev': ep_ae_rec_dev,
                    'out_dist': config.out_dist,
                    'only_AE': config.only_AE,
                    'unit_decoder_var': config.unit_decoder_var,
                    'enc_var_init': config.enc_var_init,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))
        else:
            if epoch_i == 0:
                err_p = -np.mean(val_vae_rec_losses) + np.mean(val_vae_kl_losses)
                best_model_state = model.state_dict()
            else:
                if -np.mean(val_vae_rec_losses) + np.mean(val_vae_kl_losses) > (100 - config.lr_tol) * err_p / 100:
                    logging.info(
                        "Val loss went up, Changing learning rate from {:.6f} to {:.6f}".format(lr, config.lrr * lr))
                    lr = config.lrr * lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    model.load_state_dict(best_model_state)
                else:
                    err_p = -np.mean(val_vae_rec_losses) + np.mean(val_vae_kl_losses)
                    best_model_state = model.state_dict()

            print_log = "Epoch: {:d} ((lr={:.6f})) Tr VAE Log-likelihood={:.3f} [[recon={:.3f}, kl={:.3f} ]], Var=[[enc={:.3f},dec={:.3f}]]:: Val VAE Log-likelihood={:.3f} [[recon={:.3f}, kl={:.3f} ]], Var=[[enc={:.3f},dec={:.3f}]]".format(
                epoch_i + 1, lr,
                ep_vae_rec_tr[-1] - ep_vae_kl_tr[-1], ep_vae_rec_tr[-1], ep_vae_kl_tr[-1], ep_enc_var_tr[-1],
                ep_dec_var_tr[-1],
                ep_vae_rec_dev[-1] - ep_vae_kl_dev[-1], ep_vae_rec_dev[-1], ep_vae_kl_dev[-1], ep_enc_var_dev[-1],
                ep_dec_var_dev[-1])

            logging.info(print_log)

            if (epoch_i + 1) % config.model_save_interval == 0:
                model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
                torch.save({
                    'epoch': epoch_i + 1,
                    'conf': conf,
                    'feature_dim': config.feature_dim,
                    'num_frames': num_frames,
                    'encoder_num_layers': config.encoder_num_layers,
                    'decoder_num_layers': config.decoder_num_layers,
                    'hidden_dim': config.hidden_dim,
                    'bn_dim': config.bn_dim,
                    'ep_vae_kl_tr': ep_vae_kl_tr,
                    'ep_vae_rec_tr': ep_vae_rec_tr,
                    'ep_vae_kl_dev': ep_vae_kl_dev,
                    'ep_vae_rec_dev': ep_vae_rec_dev,
                    'out_dist': config.out_dist,
                    'only_AE': config.only_AE,
                    'unit_decoder_var': config.unit_decoder_var,
                    'enc_var_init': config.enc_var_init,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


def get_ff_args():
    parser = argparse.ArgumentParser(
        description="Train a VAE")

    parser.add_argument("egs_dir", type=str, help="Path to the preprocessed data")
    parser.add_argument("store_path", type=str, help="Where to save the trained models and logs")

    parser.add_argument("--encoder_num_layers", default=3, type=int, help="Number of encoder layers")
    parser.add_argument("--decoder_num_layers", default=1, type=int, help="Number of decoder layers")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Number of hidden nodes")
    parser.add_argument("--bn_dim", default=30, type=int, help="Bottle neck dim")

    parser.add_argument("--script_path", type=str,
                        help="Path to script, if you want to save it in experiment directory")
    parser.add_argument("--config_path", type=str,
                        help="Path to config, if you want to save it in experiment directory")

    # Training configuration
    parser.add_argument("--optimizer", default="adamNoMom", type=str,
                        help="Optimizer (adam, adamNoMom, sgd")
    parser.add_argument("--batch_size", default=64, type=int, help="Training minibatch size")
    # parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--train_set", default="train_si284", help="Name of the training datatset")
    parser.add_argument("--dev_set", default="test_dev93", help="Name of development dataset")
    parser.add_argument("--adv_set", default=None, help="Full path to dev-set from other domain")
    parser.add_argument("--clip_thresh", type=float, default=1000, help="Gradient clipping threshold")

    # Misc configurations
    parser.add_argument("--feature_dim", default=13, type=int, help="The dimension of the input and predicted frame")
    parser.add_argument("--model_save_interval", type=int, default=10,
                        help="Number of epochs to skip before every model save")
    parser.add_argument("--use_gpu", action="store_true", help="Set to use GPU, code will automatically detect GPU ID")
    parser.add_argument("--load_data_workers", default=10, type=int, help="Number of parallel data loaders")
    parser.add_argument("--experiment_name", default="exp_run", type=str, help="Name of this experiment")

    # Log-variance settings
    parser.add_argument("--enc_lv", default="diagonal", type=str,
                        help="Type of encoder (log-)variance (fixed, isotropic, diagonal)")
    parser.add_argument("--dec_lv", default="diagonal", type=str,
                        help="Type of decoder (log-)variance (fixed, isotropic, diagonal)")
    parser.add_argument("--enc_lv_init_v", type=float, default=0,
                        help="Initial log-variance for encoder (value of lv when lv is fixed)")
    parser.add_argument("--dec_lv_init_v", type=float, default=0,
                        help="Initial log-variance for decoder (value of lv when lv is fixed)")
    parser.add_argument("--enc_min_lv", type=float, default=-5,
                        help="Minimum log-variance for encoder (variance is this value + predicted lv)")
    parser.add_argument("--dec_min_lv", type=float, default=-3,
                        help="Minimum log-variance for decoder (variance is this value + predicted lv)")
    parser.add_argument("--enc_shared_cov", action="store_true",
                        help="Covariance matrix for encoder will be shared (won't depend on z)")
    parser.add_argument("--dec_shared_cov", action="store_true",
                        help="Covariance matrix for decoder will be shared (won't depend on x)")
    parser.add_argument("--rotation_vae", action="store_true",
                        help="First layer in the encoder will be rotation, inverse of that rotation will be applied")

    # TODO joint vs alternate training?
    return parser.parse_args()


def runFF(config):
    experim_dir = os.path.join(config.store_path, config.experiment_name + '.dir')
    models_dir = os.path.join(experim_dir, 'models')
    plots_dir = os.path.join(experim_dir, 'plots')
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    if config.config_path:
        subprocess.run(f"cp {config.config_path} {experim_dir}", shell=True)
    if config.script_path:
        subprocess.run(f"cp {config.script_path} {experim_dir}", shell=True)

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(experim_dir, config.experiment_name),
        filemode='w')

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Load feature configuration
    egs_config = pkl.load(open(os.path.join(config.egs_dir, config.train_set, 'egs.config'), 'rb'))
    context = egs_config['concat_feats']

    num_frames = 0
    if context is not None:
        context = context.split(',')
        num_frames += int(context[0]) + int(context[1]) + 1
    else:
        num_frames += 1

    logging.info('Model Parameters: ')
    logging.info('Encoder Number of Layers: %d' % (config.encoder_num_layers))
    logging.info('Decoder Number of Layers: %d' % (config.decoder_num_layers))
    logging.info('Hidden Dimension: %d' % (config.hidden_dim))
    logging.info('Data dimension: %d' % (config.feature_dim))
    logging.info('Bottleneck dimension: %d' % (config.bn_dim))
    logging.info('Number of Frames: %d' % (num_frames))
    logging.info('Optimizer: %s ' % (config.optimizer))
    logging.info('Batch Size: %d ' % (config.batch_size))
    logging.info(f'Train set: {config.train_set} ')
    logging.info(f'Dev set: {config.dev_set} ')
    logging.info(f'Egs dir: {config.egs_dir} ')

    sys.stdout.flush()

    ## COV:
    # fixxed
    # shared / per_example_different
    # diag / isotropic / full (not supported; we would need different loss function)

    conf = DotMap()
    conf["enc"] = conf_enc = DotMap()
    conf["dec"] = conf_dec = DotMap()

    # conf["N_EPOCHS"] = 150
    # conf["N_ENC_CYCLES"] = 100
    # conf["N_DEC_CYCLES"] = 100
    # conf["BATCH_SIZE"] = 512
    DATA_DIM = conf["INPUT_DIM"] = config.feature_dim  # 80
    conf["LATENT_DIM_M"] = config.bn_dim  # 50
    conf["HID_ENC"] = conf["HID_DEC"] = config.hidden_dim  # 512
    conf["SAVE_PER_EPOCH"] = config.model_save_interval

    conf["OPTIMIZER"] = config.optimizer

    def adam(params):
        return optim.Adam(params, lr=1e-4)

    def adamNoMom(params):
        return optim.Adam(params, lr=1e-4, betas=(0, 0.9))

    def sgd(params):
        return optim.SGD(params, lr=1e-4)

    if config.optimizer in ["adam", "adamNoMom", "sgd"]:
        config.optimizer = eval(config.optimizer)
    else:
        logging.error("Wrong optimizer, use one of: adam, adamNoMom, sgd.")
        sys.exit(1)

    conf_enc["MIN_LV"] = config.enc_min_lv  # -5
    conf_dec["MIN_LV"] = config.dec_min_lv  # -3

    conf_enc["SHARED_COV"] = config.enc_shared_cov
    conf_dec["SHARED_COV"] = config.dec_shared_cov
    conf.experiment_name = config.experiment_name

    if config.enc_lv.lower() == "fixed":
        conf_enc["FIX_LV"] = True
        conf_enc["FIX_LV_CONST"] = config.enc_lv_init_v
        logging.info('Encoder lv - fixed')
    else:
        conf_enc["FIX_LV"] = False
        conf_enc["LV_INIT_V"] = config.enc_lv_init_v
        if config.enc_lv.lower() == "isotropic":
            conf_enc["out_dim_lv"] = 1
            logging.info('Encoder lv - isotropic')
        elif config.enc_lv.lower() == "diagonal":
            conf_enc["out_dim_lv"] = conf["LATENT_DIM_M"]
            logging.info('Encoder lv - diagonal')
        else:
            logging.error("Wrong choice of encoder log-variance, use one of: fixed, isotropic, diagonal.")
            sys.exit(1)
    logging.info(f'Encoder lv {"IS SHARED" if config.enc_shared_cov else "is NOT shared"}.')
    # if config.enc_shared_cov:
    #    logging.info('Encoder lv is shared.')

    if config.dec_lv.lower() == "fixed":
        conf_dec["FIX_LV"] = True
        conf_dec["FIX_LV_CONST"] = config.dec_lv_init_v
        logging.info('Decoder lv - fixed')
    else:
        conf_dec["FIX_LV"] = False
        conf_dec["LV_INIT_V"] = config.dec_lv_init_v
        if config.dec_lv.lower() == "isotropic":
            conf_dec["out_dim_lv"] = 1
            logging.info('Decoder lv - isotropic')
        elif config.dec_lv.lower() == "diagonal":
            conf_dec["out_dim_lv"] = DATA_DIM
            logging.info('Decoder lv - diagonal')
        else:
            logging.error("Wrong choice of decoder log-variance, use one of: fixed, isotropic, diagonal.")
            sys.exit(1)

    logging.info(f'Decoder lv {"IS SHARED" if config.dec_shared_cov else "is NOT shared"}.')
    # if config.dec_shared_cov:
    #    logging.info('Decoder lv is shared.')

    conf["OUTPUT_DIM_M"] = conf["INPUT_DIM"]

    # Number of hidden layers
    conf_enc["NUM_OF_LAYERS"] = config.encoder_num_layers  # 2
    conf_dec["NUM_OF_LAYERS"] = config.decoder_num_layers  # 2

    ## Clipping
    conf["CLIPPING"] = False  # TODO: try to set to true?
    conf["CLIP_VALUE"] = 100  # config.clip_thresh  # 0.2
    conf["EVAL_PER_EPOCH"] = 1

    ## Do not touch
    conf_enc["in_dim"] = conf["INPUT_DIM"]
    conf_enc["hid_dim"] = conf["HID_ENC"]
    conf_enc["out_dim_m"] = conf["LATENT_DIM_M"]
    # conf_enc["out_dim_lv"] = conf["LATENT_DIM_LV"]

    conf_dec["in_dim"] = conf["LATENT_DIM_M"]
    conf_dec["hid_dim"] = conf["HID_DEC"]
    conf_dec["out_dim_m"] = conf["OUTPUT_DIM_M"]

    device = torch.device('cuda' if config.use_gpu else 'cpu')
    conf["MODEL_SAVE_DIR"] = models_dir
    # conf["PLOTS_SAVE_DIR"] = plots_dir
    conf.rotation_vae = config.rotation_vae
    if config.rotation_vae:
        model = FF_VAE_Rotation(conf, device)
    else:
        model = FF_VAE(conf, device)

    #if config.use_gpu:
        # Set environment variable for GPU ID
        #id = get_device_id()
        #os.environ["CUDA_VISIBLE_DEVICES"] = id

    logging.info(f'config: {conf}')

    # Load Datasets
    logging.info('Loading train dataset')
    dataset_train = nnetDatasetSeqAE(os.path.join(config.egs_dir, config.train_set))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                                    num_workers=config.load_data_workers)
    logging.info('Loading val dataset')
    dataset_dev = nnetDatasetSeqAE(os.path.join(config.egs_dir, config.dev_set))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=config.batch_size, num_workers=1)
    if config.adv_set is not None:
        logging.info('Loading adv dataset')
        dataset_adv = nnetDatasetSeqAE(config.adv_set)
        data_loader_adv = torch.utils.data.DataLoader(dataset_adv, batch_size=config.batch_size, num_workers=1)
    else:
        data_loader_adv = None
    logging.info('Datasets loaded')

    train_l, val_l, dif_l = model.train_model(n_epochs=config.epochs, enc_cycles_c=0, dec_cycles_c=0,
                                              optimizer_type=config.optimizer,
                                              train_iter=data_loader_train, val_iter=data_loader_dev,
                                              adv_val_iter=data_loader_adv)

    save_fig(get_detailed_loss(train_l, "Training loss"), plots_dir, "tr_loss")
    if config.epochs > 10:
        save_fig(get_detailed_loss(train_l, "Training loss after epoch 10", 10), plots_dir, "tr_loss_ep10")
    if val_l is not None:
        save_fig(get_detailed_loss(val_l, "Validation loss"), plots_dir, "val_loss")
        if config.epochs > 10:
            save_fig(get_detailed_loss(val_l, "Validation loss after epoch 10", 10), plots_dir, "val_loss_ep10")
        if dif_l is not None:
            save_fig(get_detailed_loss(dif_l, "Difference between val-losses"), plots_dir, "diff_loss")
            if config.epochs > 10:
                save_fig(get_detailed_loss(dif_l, "Difference between val-losses after epoch 10", 10), plots_dir,
                         "diff_loss_ep10")


if __name__ == '__main__':
    config = get_args()
    run(config)
