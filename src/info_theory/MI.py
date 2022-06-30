"""
@author: samiksadhu
"""
import sys

'Generate a min-max file for binning histogram'

import argparse
import numpy as np
import bisect
import os
import pickle as pkl
from os import listdir
from os.path import join
import torch
import json

char_list = ["<blank>",
             "<unk>",
             "!",
             "\"",
             "&",
             "'",
             "(",
             ")",
             "*",
             ",",
             "-",
             ".",
             "/",
             ":",
             ";",
             "<*IN*>",
             "<*MR.*>",
             "<NOISE>",
             "<space>",
             "?",
             "A",
             "B",
             "C",
             "D",
             "E",
             "F",
             "G",
             "H",
             "I",
             "J",
             "K",
             "L",
             "M",
             "N",
             "O",
             "P",
             "Q",
             "R",
             "S",
             "T",
             "U",
             "V",
             "W",
             "X",
             "Y",
             "Z",
             "_",
             "`",
             "{",
             "}",
             "~",
             "<eos>"]


class MI:
    def __init__(self,
                 dir: list,
                 post_dir: list,
                 out_file: str,
                 data_json: list,
                 num_bins: int = 100,
                 post_type: str = 'attn'):

        self.dir = dir
        self.post_dir = post_dir
        self.post_ids = [[f for f in listdir(one_post_dir) if f.startswith('post')] for one_post_dir in self.post_dir]
        self.dir_ids = [[f for f in listdir(one_dir) if f.endswith('.pt')] for one_dir in self.dir]
        self.num_datasets = len(self.dir)
        self.lengths = [pkl.load(open(one_dir + '/lengths.pkl', 'rb')) for one_dir in self.dir]
        self.out_file = out_file
        self.data_json = [json.load(open(one_json, 'rb')) for one_json in data_json]
        combine_jsons = {}
        for one_json in self.data_json:
            combine_jsons.update(one_json['utts'])
        self.data_json = {}
        self.data_json['utts'] = combine_jsons
        self.post_type = post_type
        self.data_dim = self.get_feat_dim()
        self.post_dim = self.get_post_dim()
        self.num_bins = num_bins
        self.feat_minmax = self.get_minmax()
        self.sig_bins = None
        self.cer_bins = None
        self.cer_dict = self.compile_wer_into_dict()
        self.conf_mat = None
        self.conf_mat_post = None

    def get_feat_dim(self):
        one_id = self.dir_ids[0][0]
        X = torch.load(join(self.dir[0], one_id)).data.numpy()[:self.lengths[0][one_id]]
        return X.shape[1]

    def get_post_dim(self):
        one_id = self.post_ids[0][0]
        one_dict = pkl.load(open(join(self.post_dir[0], one_id), 'rb'))
        X = one_dict[list(one_dict.keys())[0]][self.post_type][0]
        return X.shape[1]

    def get_minmax(self):
        feat_min = +np.inf
        feat_max = -np.inf
        print('Computing min and max over the entire dataset to get bin limits')
        for j in range(self.num_datasets):
            for idx in self.dir_ids[j]:
                X = torch.load(join(self.dir[j], idx)).data.numpy()[:self.lengths[j][idx]]
                one_max = np.max(X)
                one_min = np.min(X)
                if one_max > feat_max:
                    feat_max = one_max
                if one_min < feat_min:
                    feat_min = one_min

        print('Found min value over all datasets = {:f}'.format(feat_min))
        print('Found max value over all datasets = {:f}'.format(feat_max))
        return feat_min, feat_max

    def __compute_cer(self, r, h):

        # initialisation
        import numpy
        d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
        d_sdi = numpy.zeros(((len(r) + 1), (len(h) + 1)))
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
                    d_sdi[i, j] = float(np.argmin([substitution, insertion, deletion]) + 1)

        return d, d_sdi, d[len(r)][len(h)]

    def __post2char(self, post, char_list):
        all_char = []
        post = np.argmax(post, axis=0)
        for p in post:
            all_char.append(char_list[p])
        return all_char

    def __remove_reps(self, s):
        seen = s[0]
        ans = [s[0]]
        all_idx = [0]

        for idx, i in enumerate(s[1:]):
            if i != seen:
                ans.append(i)
                all_idx.append(idx + 1)
                seen = i
        return ans, all_idx

    def compute_cer(self, key, post):
        decoded_chars = self.__post2char(post, char_list)
        token_ids = [int(x) for x in self.data_json['utts'][key]['output'][0]['tokenid'].strip().split()]
        tokens = [x for x in self.data_json['utts'][key]['output'][0]['token'].strip().split()]

        # remove blank
        decoded_chars_idx = [idx for idx, x in enumerate(decoded_chars) if x != '<blank>']
        decoded_chars = [decoded_chars[x] for x in decoded_chars_idx]
        post = post[:, decoded_chars_idx]

        # Remove repeats
        decoded_chars, decoded_chars_idx = self.__remove_reps(decoded_chars)
        post = post[:, decoded_chars_idx]

        _, _, cer = self.__compute_cer(tokens, decoded_chars)

        return cer, post

    def compile_wer_into_dict(self):
        # Compile wer
        all_wer = {}
        print('Compiling all CER into a dictionary')
        sys.stdout.flush()
        for j in range(self.num_datasets):
            for f in self.post_ids[j]:
                one_dict = pkl.load(open(join(self.post_dir[j], f), 'rb'))
                for key in one_dict:
                    one_post = np.exp(one_dict[key][self.post_type][0]).T
                    cer, _ = self.compute_cer(key, one_post)
                    all_wer[key + '.pt'] = cer

        return all_wer

    def compute_wer_confusion_matrix(self):

        self.sig_bins = np.linspace(self.feat_minmax[0], self.feat_minmax[1], self.num_bins + 1)
        self.cer_bins = np.linspace(0, 100, self.num_bins + 1)
        dist = np.zeros((self.data_dim, self.num_bins, self.num_bins))

        print('Populating the confusion matrix')
        sys.stdout.flush()
        for j in range(self.num_datasets):
            for idx in self.dir_ids[j]:
                X = torch.load(join(self.dir[j], idx)).data.numpy()[:self.lengths[j][idx]]
                jj = int(bisect.bisect_left(self.cer_bins, self.cer_dict[idx]))
                # print(self.cer_dict[idx])
                # print(jj)
                if jj == self.num_bins + 1:
                    jj = self.num_bins
                jj = jj - 1
                for t in range(X.shape[0]):
                    for r in range(self.data_dim):
                        ii = int(bisect.bisect_left(self.sig_bins, X[t, r]))

                        if ii == self.num_bins + 1:
                            ii = self.num_bins
                        ii = ii - 1
                        dist[r, ii, jj] += 1

        self.conf_mat = dist

        return dist

    def compute_wer_post_confusion_matrix(self):
        self.post_bins = np.linspace(0, 1, self.num_bins + 1)
        self.cer_bins = np.linspace(0, 100, self.num_bins + 1)
        dist = np.zeros((self.post_dim, self.num_bins, self.num_bins))

        print('Populating the confusion matrix')
        sys.stdout.flush()
        for j in range(self.num_datasets):
            for f in self.post_ids[j]:
                one_dict = pkl.load(open(join(self.post_dir[j], f), 'rb'))
                for key in one_dict:
                    one_post = np.exp(one_dict[key][self.post_type][0]).T
                    _, X = self.compute_cer(key, one_post)
                    jj = int(bisect.bisect_left(self.cer_bins, self.cer_dict[key + '.pt']))
                    if jj == self.num_bins + 1:
                        jj = self.num_bins
                    jj = jj - 1
                    for t in range(X.shape[1]):
                        for r in range(self.post_dim):
                            ii = int(bisect.bisect_left(self.post_bins, X[r, t]))
                            if ii == self.num_bins + 1:
                                ii = self.num_bins
                            ii = ii - 1
                            dist[r, ii, jj] += 1

        self.conf_mat_post = dist

        return dist

    def compute_MI(self, of_post=False):

        if of_post:
            if self.conf_mat_post is None:
                print('Confusion matrix not computed yet')
                sys.stdout.flush()
            else:
                print('Computing MI')
                sys.stdout.flush()
                MI_list = []
                for d in range(self.post_dim):
                    mat = self.conf_mat_post[d] + 1
                    num = np.sum(mat)
                    jt = mat / num
                    p_sti = np.sum(mat, axis=1) / num
                    p_sti = p_sti[None, :].T
                    p_res = np.sum(mat, axis=0) / num
                    p_res = p_res[None, :]
                    jt_indep = np.matmul(p_sti, p_res)
                    mi = np.sum(jt * np.log2(jt / jt_indep))
                    MI_list.append(mi)
        else:
            if self.conf_mat is None:
                print('Confusion matrix not computed yet')
                sys.stdout.flush()
            else:
                print('Computing MI')
                sys.stdout.flush()
                MI_list = []
                for d in range(self.data_dim):
                    mat = self.conf_mat[d] + 1
                    num = np.sum(mat)
                    jt = mat / num
                    p_sti = np.sum(mat, axis=1) / num
                    p_sti = p_sti[None, :].T
                    p_res = np.sum(mat, axis=0) / num
                    p_res = p_res[None, :]
                    jt_indep = np.matmul(p_sti, p_res)
                    mi = np.sum(jt * np.log2(jt / jt_indep))
                    MI_list.append(mi)
        return MI_list

    def compute_MI_accurate(self, of_post=False):

        if self.conf_mat is None:
            print('Confusion matrix not computed yet')
            sys.stdout.flush()
        else:
            print('Computing MI')
            sys.stdout.flush()
            MI_list = []

            # avoid nans
            for d in range(self.data_dim):
                self.conf_mat[d] += 1

            # WER distribution
            mat = self.conf_mat[0]
            num = np.sum(mat)
            jt = mat / num
            p_res_count = np.sum(mat, axis=0)
            p_res = np.sum(mat, axis=0) / num
            mi_acc = 0
            given_dists=[]
            for d in range(self.data_dim):
                mat = self.conf_mat[d]
                p_st_given_res = mat / p_res_count
                given_dists.append(p_st_given_res)
                jt = mat / num
                p_sti = np.sum(mat, axis=1) / num
                P_sti_tiled = np.tile(p_sti, (mat.shape[1], 1)).T

                mi = np.sum(p_st_given_res * np.log2(p_st_given_res / P_sti_tiled))
                mi_acc += mi
        return mi_acc, p_res, given_dists
