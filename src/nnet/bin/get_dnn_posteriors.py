import json
import os
import logging
import argparse
import subprocess
import sys
from types import SimpleNamespace
import torch
from speech_mi.model_and_data import get_model_and_sampler
from os.path import abspath, dirname
from speech_mi.utils import load_from_checkpoint, framing
from src.featgen.features import dict2Ark
import numpy as np
import pickle as pkl


def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))


def get_args():
    parser = argparse.ArgumentParser(description="Dump likelihoods or posteriors from decoding")

    parser.add_argument("model_ckpt", help="Pytorch model checkpoint")
    parser.add_argument("test_pt_file_directory", help="Test directory with .pt files")
    parser.add_argument("save_file", help="file to save posteriors")
    parser.add_argument("--prior", default=None, help="Provide prior to normalize and get likelihoods")
    parser.add_argument("--prior_weight", type=float, default=0.8, help="Weight for the prior distribution")
    parser.add_argument("--cpu", action="store_true", help="Use cpu only")
    parser.add_argument("--use_logits", action="store_true", help="Use logits instead of log-posteriors")
    return parser.parse_args()


def dump_posteriors(config):
    with open("{}/variant.json".format(dirname(abspath(config.model_ckpt)))) as json_file:
        configs = json.load(json_file)
    model_args = SimpleNamespace(**configs)
    model_args.mean_path = None
    model_args.std_path = None
    model, _ = get_model_and_sampler(model_args, config.device, False)
    load_from_checkpoint(config.model_ckpt, model, config.device)
    model = model.to(config.device)
    model.eval()

    logits2log_posterior = torch.nn.LogSoftmax(1)
    pad = torch.nn.ReflectionPad1d(model_args.context_len)
    post_dict = {}

    feature_files = [os.path.join(config.test_pt_file_directory, f) for f in os.listdir(config.test_pt_file_directory)
                     if f.endswith('.pt')]
    if config.prior:
        prior = pkl.load(open(config.prior, 'rb'))
    with torch.no_grad():
        for utt_id_pt in feature_files:
            utt_id = os.path.basename(utt_id_pt)
            utt_id = utt_id[:-3]  # The main utt_id removing .pt
            logging.info('Dumping likelihood/posterior for utt-id: {:s}'.format(utt_id))
            sys.stdout.flush()
            mat = pad(torch.load(utt_id_pt).to(config.device).T).T

            batch = framing(mat, 2 * model_args.context_len + 1).unsqueeze(1)
            #print(batch.shape)
            max_size = 1000
            act_ind = 0
            scores = []
            while act_ind < batch.size(0):
                score = model.get_logits(batch[act_ind:act_ind + max_size], normalize_input=True)
                if not config.use_logits:
                    score = logits2log_posterior(score)
                act_ind += max_size
                scores.append(score.cpu().numpy())
            scores = np.concatenate(scores)
            if config.prior:
                prior_log_wts = config.prior_weight * prior
            else:
                prior_log_wts = 0

            post_dict[utt_id] = scores - prior_log_wts

        return post_dict

def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]

if __name__ == '__main__':
    config = get_args()
    config.device = "cuda" if torch.cuda.is_available() and not config.cpu else "cpu"
    if config.device == "cuda":
        cuda_id = int(get_device_id())
        print(f"CUDA: {cuda_id}")
        torch.cuda.set_device(cuda_id)
        dummy = torch.zeros(1)
        dummy.to(config.device)

    post_dict = dump_posteriors(config)
    dict2Ark(post_dict, os.path.abspath(config.save_file), kaldi_cmd='copy-feats')
