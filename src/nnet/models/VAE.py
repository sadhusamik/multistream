import logging
import os

import torch
from bokeh.models import LegendItem, Legend
from bokeh.plotting import figure
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable
from bokeh.io import export_png, export_svgs
import numpy as np
import sys
from abc import ABCMeta, abstractmethod


class VAEEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, bn_size, dropout):
        super(VAEEncoder, self).__init__()

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.dropout = nn.Dropout(dropout)
        self.means = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=hidden_size, out_channels=bn_size, kernel_size=1, stride=1)

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):

            rnn_inputs, _ = layer(rnn_inputs)

            rnn_inputs, _ = pad_packed_sequence(
                rnn_inputs, True, total_length=seq_len)

            if i + 1 < len(self.layers):
                rnn_inputs = self.dropout(rnn_inputs)

            rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)

        means = self.means(torch.transpose(inputs, 1, 2))
        vars = self.vars(torch.transpose(inputs, 1, 2))

        return torch.transpose(means, 1, 2), torch.transpose(vars, 1, 2), inputs


class VAEDecoder(nn.Module):
    def __init__(self, bn_size, num_layers, hidden_size, input_size, unit_decoder_var=True, shared_decoder_var=False,
                 use_gpu=True):
        super(VAEDecoder, self).__init__()

        input_sizes = [bn_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in
             zip(input_sizes, output_sizes)])

        self.means = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, stride=1)
        self.vars = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, stride=1)

        self.unit_decoder_var = unit_decoder_var
        self.shared_decoder_var = shared_decoder_var
        self.use_gpu = use_gpu
        self.input_size = input_size

    def forward(self, inputs, lengths):
        seq_len = inputs.size(1)
        packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)

        for i, layer in enumerate(self.layers):
            packed_rnn_inputs, _ = layer(packed_rnn_inputs)

        inputs, _ = pad_packed_sequence(packed_rnn_inputs, True, total_length=seq_len)

        means = self.means(torch.transpose(inputs, 1, 2))
        if self.unit_decoder_var:
            if self.use_gpu:
                vars = torch.zeros((inputs.shape[0], self.input_size, inputs.shape[1])).cuda()
            else:
                vars = torch.zeros((inputs.shape[0], self.input_size, inputs.shape[1]))
        else:
            if self.shared_decoder_var:
                # Shared diagonal co-variance
                vars = self.vars(torch.ones(torch.transpose(inputs, 1, 2).shape).to(inputs.device))
            else:
                # A general diagonal co-variance
                vars = self.vars(torch.transpose(inputs, 1, 2))

        return torch.transpose(means, 1, 2), torch.transpose(vars, 1, 2)


class latentSampler(nn.Module):
    def __init__(self, use_gpu=True):
        super(latentSampler, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, latent):
        if self.use_gpu:
            return latent[0] + torch.exp(latent[1]) * torch.randn(latent[0].shape).cuda()
        else:

            return latent[0] + torch.exp(latent[1]) * torch.randn(latent[0].shape)


class nnetVAE(nn.Module):
    """
        A Variational Autoencoder (VAE) implementation in pyTorch
    """

    def __init__(self, input_size, num_layers_enc, num_layers_dec, hidden_size, bn_size,
                 dropout, use_gpu=True, only_AE=False, unit_decoder_var=True, use_transformer=False):
        super(nnetVAE, self).__init__()

        self.bn_size = bn_size
        self.gpu = use_gpu

        self.vae_encoder = VAEEncoder(input_size, num_layers_enc, hidden_size, bn_size, dropout)
        self.vae_decoder = VAEDecoder(bn_size, num_layers_dec, hidden_size, input_size,
                                      unit_decoder_var=unit_decoder_var, use_gpu=use_gpu)
        self.sampler = latentSampler(use_gpu)
        self.only_AE = only_AE
        self.unit_decoder_var = unit_decoder_var
        if only_AE:
            self.scoring_fn = self.ae_loss_score
        else:
            self.scoring_fn = self.vae_loss_score

    def forward(self, inputs, lengths):
        latent = self.vae_encoder(inputs, lengths)
        if self.only_AE:
            output = self.vae_decoder(latent[0], lengths)
            return output[0], latent
        else:
            inputs = self.sampler(latent)
            output = self.vae_decoder(inputs, lengths)
            return output, latent

    def init_enc_var(self, logvar_value=-25):
        with torch.no_grad():
            self.vae_encoder.vars.bias.data.add_(logvar_value)

    def __pad2list(self, padded_seq, lengths):
        """
        Concatenate all the examples in a batch and remove the appended zeros at the end
        :param padded_seq:
        :param lengths:
        :return:
        """
        return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])

    def ae_loss(self, x, ae_out, out_dist='gauss'):
        """

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param ae_out:  Tensor (Batch x num_frames x data_dimensions)
        :param out_dist: str ('gauss' or 'laplace')
        :return:
        """
        if out_dist == 'gauss':
            llhood = torch.sum(-0.5 * torch.pow((x - ae_out), 2) - 0.5 * np.log(2 * np.pi * 1), dim=2)
        elif out_dist == 'laplace':
            llhood = torch.sum(-torch.abs(x - ae_out) - np.log(2), dim=2)

        else:
            print("Output distribution of VAE can be 'gauss' or 'laplace'")
            sys.exit(1)

        return llhood

    def ae_loss_train(self, x, batch_l, ae_out, out_dist='gauss'):
        """
        Used to compute autoencoder loss during training

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param batch_l: Tensor (Batch)
        :param latent_out: Tensor (Batch x num_frames x latent_dimensions)
        :param out_dist: out_dist: str ('gauss' or 'laplace')
        :return:
        """
        x = self.__pad2list(x, batch_l).unsqueeze(0)
        ae_out = self.__pad2list(ae_out, batch_l).unsqueeze(0)
        llhood = self.ae_loss(x, ae_out, out_dist)

        return torch.mean(llhood)

    def ae_loss_score(self, x, ae_out, latent_out, out_dist='gauss', temp=1, type='prod', chop=None, vad_idx=None):
        """

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param ae_out:  Tensor (Batch x num_frames x data_dimensions)
        :param latent_out:  Tensor (Batch x num_frames x latent_dimensions)
        :param out_dist: str ('gauss' or 'laplace')
        :param temp: float (Inverse temperature of softmax)
        :param type: str (Way to combine frame-wise likelihood 'prod' or 'sum' )
        :param chop: float (useless option)
        :return:
        """

        import copy
        if chop:
            num_fr = x.shape[1]
            perc = int(np.ceil(chop * 100))
            selection = np.random.randint(0, num_fr, perc)

            # Chop the utterances
            x = x[:, selection, :]
            ae_out = ae_out[:, selection, :]

        log_lhood = self.ae_loss(x, ae_out, out_dist)
        score = log_lhood[0]

        if vad_idx:
            score = score[vad_idx]

        llhood = copy.deepcopy(score.data.numpy())
        if type == 'prod':
            score = temp * torch.mean(score)
        elif type == 'sum':
            score = torch.log(torch.sum(torch.exp(temp * score)))
        else:
            raise ValueError("Score type can only be 'sum' or 'prod'")

        return score, llhood

    def vae_loss_train(self, x, batch_l, ae_out, latent_out, out_dist='gauss'):
        """
        Used to compute VAE loss during training

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param batch_l: Tensor (Batch)
        :param ae_out: Tensor (Batch x num_frames x data_dimensions)
        :param latent_out: Tensor (Batch x num_frames x latent_dimensions)
        :param out_dist: out_dist: str ('gauss' or 'laplace')
        :return:
        """
        x = self.__pad2list(x, batch_l).unsqueeze(0)
        ae_out = (self.__pad2list(ae_out[0], batch_l).unsqueeze(0), self.__pad2list(ae_out[1], batch_l).unsqueeze(0))
        latent_out = (
            self.__pad2list(latent_out[0], batch_l).unsqueeze(0), self.__pad2list(latent_out[1], batch_l).unsqueeze(0))
        log_lhood, kl_loss = self.vae_loss(x, ae_out, latent_out, out_dist)

        return torch.mean(log_lhood), torch.mean(kl_loss)

    def vae_loss_score(self, x, ae_out, latent_out, out_dist='gauss', temp=1, type='prod', chop=None, vad_idx=None):
        """

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param ae_out (Tuple):  (Mean -> Tensor (Batch x num_frames x data_dimensions), Vars- > Tensor (Batch x num_frames x data_dimensions))
        :param latent_out:  Tensor (Batch x num_frames x latent_dimensions)
        :param out_dist: str ('gauss' or 'laplace')
        :param temp: float (Inverse temperature of softmax)
        :param type: str (Way to combine frame-wise likelihood 'prod' or 'sum' )
        :param chop: float (useless option)
        :return:
        """

        import copy
        if chop:
            num_fr = x.shape[1]
            perc = int(np.ceil(chop * 100))
            selection = np.random.randint(0, num_fr, perc)

            # Chop the utterances
            x = x[:, selection, :]
            ae_out = ae_out[:, selection, :]
            latent_out = (latent_out[0][:, selection, :], latent_out[1][:, selection, :])

        log_lhood, kl_loss = self.vae_loss(x, ae_out, latent_out, out_dist)
        # if torch.abs(torch.mean(log_lhood)) > 100:
        #    log_lhood += 200  # This is to make the softmax stable and not become nan
        score = log_lhood[0] - kl_loss[0]
        llhood = copy.deepcopy(score.data.numpy())

        if vad_idx:
            score = score[vad_idx]

        if type == 'prod':
            score = temp * torch.mean(score)
        elif type == 'sum':
            score = torch.log(torch.sum(torch.exp(temp * score)))
        else:
            raise ValueError("Score type can only be 'sum' or 'prod'")

        return score, llhood

    def vae_loss(self, x, ae_out, latent_out, out_dist='gauss'):
        """

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param ae_out:  Tensor (Batch x num_frames x data_dimensions)
        :param latent_out:  Tensor (Batch x num_frames x latent_dimensions)
        :param out_dist: str ('gauss' or 'laplace')
        :return: Tuple (log_lhood, kl_loss), each (Batch x num_frames)
        """

        if out_dist == 'gauss':
            log_lhood = torch.sum(
                -0.5 * torch.pow((x - ae_out[0]), 2) / torch.pow(torch.exp(ae_out[1]), 2) - 0.5 * np.log(
                    2 * np.pi * 1) - ae_out[1], dim=2)
        elif out_dist == 'laplace':
            log_lhood = torch.sum(-torch.abs(x - ae_out[0]) / torch.exp(ae_out[1]) - np.log(2) - ae_out[1], dim=2)
        else:
            print("Output distribution of VAE can be 'gauss' or 'laplace'")
            sys.exit(1)

        kl_loss = 0.5 * torch.sum(
            1 - torch.pow(latent_out[0], 2) - torch.pow(torch.exp(latent_out[1]), 2) + 2 * latent_out[1], dim=2)
        return log_lhood, -kl_loss

    def compute_llhood(self, inputs, lengths, sample_num=10, out_dist='gauss'):
        """

        :param inputs:
        :param lengths:
        :param sample_num:
        :param out_dist:
        :return:
        """
        latent = self.vae_encoder(inputs, lengths)
        latent_out = (latent[0][0], latent[1][0])
        loss_acc_recon = 0
        loss_acc_kl = 0
        for i in range(sample_num):
            z = self.sampler(latent)
            loss = self.vae_loss(inputs[0], self.vae_decoder(z, lengths)[0], latent_out, out_dist)
            loss_acc_recon += loss[0].item()
            loss_acc_kl -= loss[1].item()

        return loss_acc_recon / sample_num, loss_acc_kl / sample_num

    def generate(self, size=512):
        """

        :param size:
        :return:
        """
        if self.gpu:
            input = torch.randn([1, size, self.bn_size]).cuda()
        else:
            input = torch.randn([1, size, self.bn_size])

        return self.vae_decoder(input, torch.IntTensor([size]))[0]


class VAE_LSTM(nnetVAE):
    """
        A Variational Autoencoder (VAE) implementation with LSTM encoder and decoder

        Config dictionary conf should have

        Required -->
        input_size: int
        num_layers_enc: int
        num_layers_dec: int
        hidden_size: int
        bn_size: int

        Optional -->
        dropout: float =0
        out_dist: str = 'gauss' ('laplace' or 'gauss')
        enc_var_init: float =0,
        only_AE=False: bool ( set to train only an autoencoder)
        unit_decoder_var=False: bool ( set to have unit decoder variance)
        shared_decoder_var=False: bool ( set to have shared decoder covariance matrix)
        use_transformer=False: bool ( [NOT IMPLEMENTED YET] set to use transformer layers)

        NOTE: Only diagonal covariance matrix is supported

    def __init__(self, conf, use_gpu=True):
        super(VAE_LSTM, self).__init__(conf, use_gpu=True)
    """

    def __init__(self, conf, use_gpu=True):
        super(nnetVAE, self).__init__()

        # Assign the optional arguments
        self.conf = {'dropout': 0, 'out_dist': 'gauss', 'enc_var_init': 0, 'only_AE': False,
                         'unit_decoder_var': False, 'shared_decoder_var': False,
                         'use_transformer': False}
        self.conf.update(conf)
        self.gpu = use_gpu
        self.bn_size = conf['bn_size']
        self.only_AE = conf['only_AE']
        self.unit_decoder_var = conf['unit_decoder_var']
        self.shared_decoder_var = conf['shared_decoder_var']
        print('Model_config:')
        print(self.conf)

        if self.only_AE:
            self.scoring_fn = self.ae_loss_score
        else:
            self.scoring_fn = self.vae_loss_score

        # Define the model
        self.vae_encoder = VAEEncoder(conf['input_size'], conf['num_layers_enc'], conf['hidden_size'], conf['bn_size'],
                                      conf['dropout'])
        self.vae_decoder = VAEDecoder(conf['bn_size'], conf['num_layers_dec'], conf['hidden_size'], conf['input_size'],
                                      unit_decoder_var=conf['unit_decoder_var'],
                                      shared_decoder_var=conf['shared_decoder_var'], use_gpu=use_gpu)
        self.sampler = latentSampler(use_gpu)

    def vae_loss_train(self, x, batch_l, ae_out, latent_out):
        """
        Used to compute VAE loss during training

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param batch_l: Tensor (Batch)
        :param ae_out: Tensor (Batch x num_frames x data_dimensions)
        :param latent_out: Tensor (Batch x num_frames x latent_dimensions)
        :return:
        """
        out_dist = self.conf['out_dist']
        x = self.__pad2list(x, batch_l).unsqueeze(0)
        ae_out = (self.__pad2list(ae_out[0], batch_l).unsqueeze(0), self.__pad2list(ae_out[1], batch_l).unsqueeze(0))
        latent_out = (
            self.__pad2list(latent_out[0], batch_l).unsqueeze(0), self.__pad2list(latent_out[1], batch_l).unsqueeze(0))
        log_lhood, kl_loss = self.vae_loss(x, ae_out, latent_out, out_dist)

        return torch.mean(log_lhood), torch.mean(kl_loss)

    def vae_loss_score(self, x, ae_out, latent_out, temp=1, type='prod', chop=None, vad_idx=None):
        """

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param ae_out (Tuple):  (Mean -> Tensor (Batch x num_frames x data_dimensions), Vars- > Tensor (Batch x num_frames x data_dimensions))
        :param latent_out:  Tensor (Batch x num_frames x latent_dimensions)
        :param temp: float (Inverse temperature of softmax)
        :param type: str (Way to combine frame-wise likelihood 'prod' or 'sum' )
        :param chop: float (useless option)
        :return:
        """
        out_dist = self.conf['out_dist']
        import copy
        if chop:
            num_fr = x.shape[1]
            perc = int(np.ceil(chop * 100))
            selection = np.random.randint(0, num_fr, perc)

            # Chop the utterances
            x = x[:, selection, :]
            ae_out = ae_out[:, selection, :]
            latent_out = (latent_out[0][:, selection, :], latent_out[1][:, selection, :])

        log_lhood, kl_loss = self.vae_loss(x, ae_out, latent_out, out_dist)
        # if torch.abs(torch.mean(log_lhood)) > 100:
        #    log_lhood += 200  # This is to make the softmax stable and not become nan
        score = log_lhood[0] - kl_loss[0]
        llhood = copy.deepcopy(score.data.numpy())

        if vad_idx:
            score = score[vad_idx]

        if type == 'prod':
            score = temp * torch.mean(score)
        elif type == 'sum':
            score = torch.log(torch.sum(torch.exp(temp * score)))
        else:
            raise ValueError("Score type can only be 'sum' or 'prod'")

        return score, llhood

    def compute_encoder_and_decoder_vars(self, x, batch_l, ae_out, latent_out):
        """
        Compute encoder and decoder variances over one batch

        :param x: Tensor (Batch x num_frames x data_dimensions)
        :param batch_l: Tensor (Batch)
        :param ae_out: Tensor (Batch x num_frames x data_dimensions)
        :param latent_out: Tensor (Batch x num_frames x latent_dimensions)
        :return:
        """
        x = self.__pad2list(x, batch_l).unsqueeze(0)
        ae_out = (self.__pad2list(ae_out[0], batch_l).unsqueeze(0), self.__pad2list(ae_out[1], batch_l).unsqueeze(0))
        latent_out = (
            self.__pad2list(latent_out[0], batch_l).unsqueeze(0), self.__pad2list(latent_out[1], batch_l).unsqueeze(0))

        var_enc = torch.mean(torch.pow(torch.exp(latent_out[1]), 2)).item()
        var_dec = torch.mean(torch.pow(torch.exp(ae_out[1]), 2)).item()

        return var_enc, var_dec

    def __pad2list(self, padded_seq, lengths):
        """
        Concatenate all the examples in a batch and remove the appended zeros at the end
        :param padded_seq:
        :param lengths:
        :return:
        """
        return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


class EncDec(nn.Module, metaclass=ABCMeta):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf
        self.device = device

        @abstractmethod
        def mean_f(self, x):
            return NotImplemented

        @abstractmethod
        def lv_f(self, x):
            return NotImplemented

        @abstractmethod
        def hid_f(self, x):
            return NotImplemented

    def forward(self, x):
        h = self.hid_f(x)
        mean = self.mean_f(h)

        if self.conf.FIX_LV:
            # print((len(x), self.conf.out_dim_m))
            return mean, torch.full((len(x), self.conf.out_dim_m), self.conf.FIX_LV_CONST, dtype=torch.double,
                                    device=self.device)

        if self.conf.SHARED_COV:
            # Masks hidden state
            h = torch.ones_like(h)
        logvar = self.lv_f(h)

        if self.conf.out_dim_lv == 1:  # Isotropic covariance matrix
            # logvar = torch.cat(self.conf.out_dim_m * [logvar], dim=-1)
            logvar = logvar.repeat(1, self.conf.out_dim_m)

        return mean, logvar


class EncDecNN(EncDec):
    def __init__(self, conf, device):
        super().__init__(conf, device)
        self.fc_m = nn.Linear(conf.hid_dim[conf.NUM_OF_LAYERS - 1], conf.out_dim_m, device=self.device)
        if not self.conf.FIX_LV:
            self.fc_lv = nn.Linear(conf.hid_dim[conf.NUM_OF_LAYERS - 1], conf.out_dim_lv, device=self.device)
        self.min_lv = np.exp(conf.MIN_LV)  # .007
        if conf.LV_INIT_V:
            with torch.no_grad():
                self.fc_lv.bias.data.add_(conf.LV_INIT_V)
            print(self.fc_lv.bias)

    def mean_f(self, x):
        return self.fc_m(x)

    def lv_f(self, x):
        return torch.log(torch.exp(self.fc_lv(x)) + self.min_lv)


class EncDecFeedForw(EncDecNN):
    def __init__(self, conf, act_fun, device):
        super().__init__(conf, device)
        self.act_fun = act_fun
        self.layer_c = conf.NUM_OF_LAYERS

        self.fcs = [nn.Linear(conf.in_dim, conf.hid_dim[0], device=self.device)]
        for i in range(1, conf.NUM_OF_LAYERS):
            self.fcs.append(nn.Linear(conf.hid_dim[i - 1], conf.hid_dim[i], device=self.device))
        self.fcs = nn.ModuleList(self.fcs)

    def hid_f(self, x):
        h = x
        for i in range(self.layer_c):
            h = self.act_fun(self.fcs[i](h))
        return h


def compute_rec(x, mu_x, logvar_x, sum_over_batch=True):
    # reconstruct_l = -log(N(x|mu_x,var_x)
    if sum_over_batch:
        return 0.5 * (torch.sum((x - mu_x).pow(2) * torch.exp(-logvar_x))
                      + torch.sum(logvar_x) + x.numel() * np.log(2 * np.pi))
    return 0.5 * (torch.sum((x - mu_x).pow(2) * torch.exp(-logvar_x), dim=1)
                  + torch.sum(logvar_x + np.log(2 * np.pi), dim=1))


def compute_kld(mu_z, logvar_z, sum_over_batch=True):
    if sum_over_batch:
        return -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    return -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class FF_VAE(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        # TODO: make it as option?
        act_fun = nn.PReLU(device=device)  # torch.relu
        # act_fun = torch.tanh
        conf["ACT_FUN"] = "prelu"  # "relu"

        self.conf = conf
        self.device = device
        if np.isscalar(conf.enc.hid_dim):
            conf.enc.hid_dim = conf.enc.NUM_OF_LAYERS * [conf.enc.hid_dim]
        if np.isscalar(conf.dec.hid_dim):
            conf.dec.hid_dim = conf.dec.NUM_OF_LAYERS * [conf.dec.hid_dim]
        self.encoder = EncDecFeedForw(conf.enc, act_fun, device)
        self.decoder = EncDecFeedForw(conf.dec, act_fun, device)

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        # logvar_z = torch.clamp(logvar_z, self.conf.MIN_LV)
        z = reparameterize(mu_z, logvar_z)

        mu_x, logvar_x = self.decoder(z)
        # logvar_x = torch.clamp(logvar_x, self.conf.MIN_LV)
        return mu_z, logvar_z, mu_x, logvar_x

    def detect_inf_and_nans(self, var, var_name):
        nan_c = torch.sum(torch.isnan(var)).item()
        inf_c = torch.sum(torch.isinf(var)).item()
        if nan_c > 0 or inf_c > 0:
            logging.error(f"Detected {nan_c} NANs and {inf_c} infs for {var_name}.")
            if var.dim() == 0:
                logging.error(var)
            else:
                logging.error(var[:2])
            logging.error(f"mu_z: {self.last_mu_z[:2]}")
            logging.error(f"logvar_z: {self.last_logvar_z[:2]}")
            logging.error(f"mu_x: {self.last_mu_x[:2]}")
            logging.error(f"logvar_x: {self.last_logvar_x[:2]}")
            logging.error(f"reconstruct_l: {self.last_reconstruct_l}")
            logging.error(f"kld: {self.last_kld}")
            sys.exit(1)

    def compute_loss(self, x, mu_z, logvar_z, mu_x, logvar_x):
        # while torch.min(logvar_x) < self.conf.MIN_LV:
        #    logvar_x = torch.where(logvar_x < self.conf.MIN_LV, 0.8 * logvar_x, logvar_x)
        # logvar_x = torch.clamp(logvar_x, self.conf.MIN_LV)
        self.detect_inf_and_nans(mu_z, "mu_z")
        self.detect_inf_and_nans(logvar_z, "logvar_z")
        self.detect_inf_and_nans(mu_x, "mu_x")
        self.detect_inf_and_nans(logvar_x, "logvar_x")

        reconstruct_l = compute_rec(x, mu_x, logvar_x)
        self.detect_inf_and_nans(reconstruct_l, "reconstruct_l")
        kld = compute_kld(mu_z, logvar_z)
        self.detect_inf_and_nans(kld, "kld")

        self.last_mu_z = mu_z
        self.last_logvar_z = logvar_z
        self.last_mu_x = mu_x
        self.last_logvar_x = logvar_x
        self.last_reconstruct_l = reconstruct_l
        self.last_kld = kld

        return reconstruct_l + kld, reconstruct_l, kld

    def compute_llhood_l(self, x, lengths=None, lat_sam_num=10, sum_over_batch=True):
        if lengths is not None:
            x = self.__pad2list(x, lengths)
        return self.compute_llhood(x, lat_sam_num, sum_over_batch)

    def compute_llhood(self, x, lat_sam_num, sum_over_batch=True):
        mu_z, logvar_z = self.encoder(x)

        reconstruct_l = 0.
        for s_num in range(lat_sam_num):
            z = reparameterize(mu_z, logvar_z)
            mu_x, logvar_x = self.decoder(z)
            # logvar_x = torch.clamp(logvar_x, self.conf.MIN_LV)
            reconstruct_l += compute_rec(x, mu_x, logvar_x, sum_over_batch)

        return -(reconstruct_l / lat_sam_num + compute_kld(mu_z, logvar_z, sum_over_batch))

    def vae_loss_score(self, x, lat_sam_num=10):
        import copy
        llhood = self.compute_llhood(x, lat_sam_num, False)
        #llhood = copy.deepcopy(score.data.numpy())
        return torch.mean(llhood), llhood

    def generate(self, batch):
        return self.decoder(torch.randn([batch, self.conf.dec.in_dim], device=self.device))

    # Reconstruction + KLD losses summed over all dimensions and batch
    def train_one_epoch(self, optimizer, data_iter):
        # print(self.encoder.fc_lv.bias)
        # print(self.decoder.fc_lv.bias)
        self.train()
        tot_rec_l = 0.
        tot_kld_l = 0.
        tot_frame_c = 0.
        tot_loss = 0.
        # for data, ind in data_iter:
        first_batch = True
        for batch_x, batch_l in data_iter:
            # data, ind = data

            # x = self.__pad2list(x, batch_l).unsqueeze(0)
            # ae_out = self.__pad2list(ae_out, batch_l).unsqueeze(0)
            # loss = self.ae_loss(x, ae_out, out_dist)

            # batch_x = torch.from_numpy(batch_x).to(self.device)
            # batch_l = torch.from_numpy(batch_l).to(self.device)
            batch_x = batch_x.to(self.device)
            batch_l = batch_l.to(self.device)
            frame_c = torch.sum(batch_l).item()

            data = self.__pad2list(batch_x, batch_l)
            # data += 0.01 * torch.randn_like(data)
            # print(batch_x.shape)
            # print(batch_l.shape)

            optimizer.zero_grad()
            mu_z, logvar_z, mu_x, logvar_x = self(data)
            # loss = sum(loss_function(recon_batch, data, mu, logvar))
            loss, rec_l, kld_l = self.compute_loss(data, mu_z, logvar_z, mu_x, logvar_x)

            if first_batch and self.conf.dec.SHARED_COV and not self.conf.dec.FIX_LV:
                logging.info(f"LOGVAR mean {logvar_x.mean().item()}, LOGVAR std {logvar_x.std().item()}")
            first_batch = False

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.error("Skipping batch due to inf or nan!")
                continue

            MAX_LOSS = 500
            if loss.item() / frame_c > MAX_LOSS:
                old_loss = loss.item()
                # loss *= (MAX_LOSS * frame_c) / old_loss
                logging.error(f"Changing loss from {old_loss / frame_c} to {loss.item() / frame_c}")
                # logging.error(f"Skipping batch due to large loss! {loss.item()/frame_c}")
                # continue

            loss.backward()
            if self.conf.CLIPPING:
                torch.nn.utils.clip_grad_value_(self.parameters(), self.conf.CLIP_VALUE)

            tot_frame_c += frame_c
            tot_rec_l += rec_l.item()
            tot_kld_l += kld_l.item()
            tot_loss += loss.item()
            optimizer.step()
        return tot_loss / tot_frame_c, tot_rec_l / tot_frame_c, tot_kld_l / tot_frame_c

    def eval_model(self, data_iter):
        self.eval()

        tot_rec_l = 0.
        tot_kld_l = 0.
        tot_loss = 0.
        tot_frame_c = 0.

        with torch.no_grad():
            for batch_x, batch_l in data_iter:
                batch_x = batch_x.to(self.device)
                batch_l = batch_l.to(self.device)
                tot_frame_c += torch.sum(batch_l).item()
                data = self.__pad2list(batch_x, batch_l)

                mu_z, logvar_z, mu_x, logvar_x = self(data)
                loss, rec_l, kld_l = self.compute_loss(data, mu_z, logvar_z, mu_x, logvar_x)
                tot_rec_l += rec_l.item()
                tot_kld_l += kld_l.item()
                tot_loss += loss.item()
        # l = data_iter.size
        return tot_loss / tot_frame_c, tot_rec_l / tot_frame_c, tot_kld_l / tot_frame_c

    def train_model(self, n_epochs, enc_cycles_c, dec_cycles_c, optimizer_type, train_iter, val_iter=None,
                    adv_val_iter=None,
                    modify_during_training=lambda *args: None):
        torch.set_printoptions(profile="full")
        # Put as self so optimizers will be saved automatically
        optimizer_all = optimizer_type(self.parameters())
        optimizer_enc = optimizer_type(self.encoder.parameters())
        optimizer_dec = optimizer_type(self.decoder.parameters())
        # self.save(0, f"model_epoch_{0}", optimizer_all)
        train_loss = []
        val_losses, diff_val_losses = [], []
        best_val_loss = 10000000
        train_jointly = enc_cycles_c == 0 and dec_cycles_c == 0
        for epoch in range(n_epochs):
            modify_during_training(self, epoch, train_iter)

            if train_jointly:  # Train jointly
                train_loss.append(self.train_one_epoch(optimizer_all, train_iter))
            else:
                for enc_cycle in range(enc_cycles_c):
                    # optimizer_enc = optimizer_type(self.encoder.parameters())
                    train_loss.append(self.train_one_epoch(optimizer_enc, train_iter))
                    # model.train_one_epoch(optimizer_enc, data)
                for dec_cycle in range(dec_cycles_c):
                    # optimizer_dec = optimizer_type(self.decoder.parameters())
                    train_loss.append(self.train_one_epoch(optimizer_dec, train_iter))

            print_log = f"Ep: {epoch + 1} Tr: {train_loss[-1][0]:.2f} {train_loss[-1][1]:.2f} {train_loss[-1][2]:.2f}"
            # Get validation loss
            if val_iter is not None and epoch % self.conf.EVAL_PER_EPOCH == self.conf.EVAL_PER_EPOCH - 1:
                val_loss = self.eval_model(val_iter)
                val_losses.append(val_loss)
                print_log += f" # V: {val_loss[0]:.2f} {val_loss[1]:.2f} {val_loss[2]:.2f}"
                if adv_val_iter is not None:
                    diff_val_loss = [a - b for a, b in zip(self.eval_model(adv_val_iter), val_loss)]
                    diff_val_losses.append(diff_val_loss)
                    print_log += f" # D: {diff_val_loss[0]:.2f} {diff_val_loss[1]:.2f} {diff_val_loss[2]:.2f}"
                if val_loss[0] < best_val_loss:
                    best_val_loss = val_loss[0]
                    logging.info(f"New best val_loss: {best_val_loss:.3f}")
                    if train_jointly:
                        self.save(epoch, f"best_val", optimizer_all)
                    else:
                        self.save(epoch, f"best_val", optimizer_enc, optimizer_dec)
            logging.info(print_log)
            if epoch % self.conf.SAVE_PER_EPOCH == self.conf.SAVE_PER_EPOCH - 1:
                if train_jointly:
                    # self.save(epoch, f"{self.conf.experiment_name}_epoch_{epoch + 1}", optimizer_all)
                    self.save(epoch, f"model_epoch_{epoch + 1}", optimizer_all)
                else:
                    # self.save(epoch, f"{self.conf.experiment_name}_epoch_{epoch + 1}", optimizer_enc, optimizer_dec)
                    self.save(epoch, f"model_epoch_{epoch + 1}", optimizer_enc, optimizer_dec)

        val_losses = None if val_losses == [] else val_losses
        diff_val_losses = None if diff_val_losses == [] else diff_val_losses
        return train_loss, val_losses, diff_val_losses

    def save(self, epoch, name, optimizer, optimizer2=None):
        d = {"epoch": epoch, "model_state_dict": self.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "config": self.conf}
        if optimizer2 is not None:
            d["optimizer2_state_dict"] = optimizer2.state_dict()
        torch.save(d, os.path.join(self.conf.MODEL_SAVE_DIR, name + '.pt'))

    def __pad2list(self, padded_seq, lengths):
        """
        Concatenate all the examples in a batch and remove the appended zeros at the end
        :param padded_seq:
        :param lengths:
        :return:
        """
        return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])


class FF_VAE_Rotation(FF_VAE):
    def __init__(self, conf, device, rotation_weight=5000.0):
        super().__init__(conf, device)
        self.rotation = nn.Linear(conf.enc.in_dim, conf.enc.in_dim, False, device=self.device)
        self.rotation.weight.data.copy_(torch.eye(conf.enc.in_dim))
        self.rotation_weight = rotation_weight
        self.rotation_loss = nn.MSELoss(reduction="mean")
        self.identity = torch.eye(conf.enc.in_dim, device=self.device)

    def forward(self, x):
        return super().forward(self.rotation(x))

    def compute_loss(self, x, mu_z, logvar_z, mu_x, logvar_x):
        rec_and_kld, reconstruct_l, kld = super().compute_loss(self.rotation(x), mu_z, logvar_z, mu_x, logvar_x)
        rotation_loss = self.rotation_weight * self.rotation_loss(
            self.rotation.weight.mm(self.rotation.weight.T), self.identity)
        return rotation_loss + rec_and_kld, reconstruct_l, kld

    def compute_llhood(self, x, lat_sam_num, sum_over_batch=True):
        return super().compute_llhood(self.rotation(x), lat_sam_num, sum_over_batch)

    def generate(self, batch):
        super().generate(batch).mm(self.rotation.weight)

    def train_one_epoch(self, optimizer, data_iter):
        print((self.rotation_weight * self.rotation_loss(
            self.rotation.weight.mm(self.rotation.weight.T), self.identity)).item())
        # print(self.rotation.weight.data)
        print(torch.linalg.matrix_rank(self.rotation.weight.data))
        return super().train_one_epoch(optimizer, data_iter)


def save_fig(fig, directory, name, sub_figs=None):
    export_png(fig, filename=os.path.join(directory, name + ".png"))
    if sub_figs is None:
        fig.output_backend = "svg"
        export_svgs(fig, filename=os.path.join(directory, name + ".svg"))
    else:
        # Bokeh does not support saving plot with more figures in svg format
        for i, sf in enumerate(sub_figs):
            sf.output_backend = "svg"
            export_svgs(sf, filename=os.path.join(directory, name + str(i + 1) + ".svg"))


def get_detailed_loss(loss, title, leave_epochs=0):
    loss, rec_l, kld_l = np.array(loss[leave_epochs:]).T
    fig = figure(title=title)
    indices = list(range(1, len(loss) + 1))
    x = [indices, indices, indices]
    y = [loss, rec_l, kld_l]
    colors = ["black", "red", "green"]

    r = fig.multi_line(x, y, color=colors)
    legends = [LegendItem(label="Rec + KLD", renderers=[r], index=0),
               LegendItem(label="Reconstr", renderers=[r], index=1), LegendItem(label="KLD", renderers=[r], index=2)]

    legend = Legend(items=legends)
    fig.add_layout(legend)
    fig.xaxis.axis_label = 'Epoch'
    fig.yaxis.axis_label = 'Loss'
    return fig

#
# class LogVarClamp(Function):
#    @staticmethod
#    def forward(ctx, i, min_val, max_val):
#        ctx._mask = i.ge(min_val)
#        return i.clamp(min=min_val)
#
#    @staticmethod
#    def backward(ctx, grad_output):
#        mask = Variable(ctx._mask.type_as(grad_output.data))
#        return grad_output * mask, None, None
