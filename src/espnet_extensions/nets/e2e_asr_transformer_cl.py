# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""
import sys
from argparse import Namespace
import logging
import math

from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

def reduce_len_per_frame(x):
    x = x[:, :-3]
    #return x.reshape(x.size(0), x.size(1)//4, 4).mean(-1)
    return x[:, :(x.size(1)//4)*4].reshape(x.size(0), -1, 4).mean(-1)

def super_ugly_hack_how_to_get_logits_from_one_specific_model(self, tgt, tgt_mask, memory, cache=None):
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        return self.output_layer(y)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module with several parallel ASR models.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(np.prod(self.subsample))

    def __init__(self, all_models, all_pms, all_enc_pms=None, all_enc_transform=None, ignore_id=-1):
        """Second level E2E object combining multiple E2E models.
        Will assume that all the parallem models are pure CTC or multitask,
        but not combinations of CTC and multitask

        :param list all_models: List of all transformer models to be used in parallel
        :param list all_pms: List of all performance monitors for model combination
        :param list all_enc_transform: List of all transforms applied on ASR encoder output before putting into VAE, can be PCA/ CMVN
        """
        torch.nn.Module.__init__(self)

        self.all_models = all_models
        self.all_pms = all_pms
        self.all_enc_pms = all_enc_pms
        self.all_enc_transform = all_enc_transform
        if all_pms:
            assert len(all_models) == len(all_pms)
        if all_enc_pms:
            assert len(all_models) == len(all_enc_pms)
            assert all_enc_transform is not None
            assert len(all_enc_transform) == len(all_enc_pms)

        self.stream_num = len(all_models)

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def sym_kld(self, X, Y):
        return torch.sum(X * (torch.log(X) - torch.log(Y))) / X.size(0) + torch.sum(
            Y * (torch.log(Y) - torch.log(X))) / X.size(0)

    def mmeasure_loss(self, X, del_list=[1, 2, 3, 4, 5, 6, 7], use_gpu=False):

        kld = nn.KLDivLoss()
        if use_gpu:
            m_acc = torch.FloatTensor([0]).cuda()
        else:
            m_acc = torch.FloatTensor([0])

        for d in del_list:
            m_acc += self.sym_kld(X[d:, :], X[0:-d, :]) + kld(X[0:-d:, :], X[d:, :])
        return m_acc / len(del_list)

    def update_temp(self, temp, wts, A, B, lr=0.01):
        A = np.asarray([np.mean(x) for x in A])
        S1 = np.sum(wts)
        S2 = np.sum(A * wts)
        S3 = 0 * B[0]
        S4 = 0 * B[0]
        tempp = A * wts
        for idx, x in enumerate(B):
            S3 += wts[idx] * x
            S4 += tempp[idx] * x

        grad_acc = np.mean(S4 / S1 - S2 * S3 / (S1 ** 2))
        logging.info("computed temperature gradient is {:f}".format(grad_acc))
        logging.info("Changing temperature from {:f} to {:f}".format(temp, temp - lr * grad_acc))
        return temp - lr * grad_acc

    def apply_cmvn(self, X, means, vars):
        return (X - torch.as_tensor(means)) / torch.as_tensor(np.sqrt(vars))

    def apply_pca(self, X, mat):
        Y = torch.transpose(torch.matmul(torch.as_tensor(mat), torch.transpose(X[0], 0, 1)), 0, 1)
        return Y.unsqueeze(0)

    def get_sc_f(self, x, enc_x, pm_type, pm_models, vae_inp_f, recog_args):
        all_llhoods = []
        sm = torch.nn.Softmax(dim=0)
        if pm_type in ["vae", "vae_ff", "vae_old"]:
            pm_scores = torch.zeros(self.stream_num)
            for idx, pm_model in enumerate(pm_models):
                inp = vae_inp_f(x, enc_x, idx)
                # TODO: forward pass through VAE
                if pm_type == "vae_ff":
                    pm_scores[idx], llhood = pm_model.vae_loss_score(inp[0])
                elif pm_type == "vae" or pm_type == "vae_old":
                    if recog_args.auto_vad:
                        eng = np.mean(inp[0] ** 2, axis=1)
                        vad_idx = [idx for idx, v in enumerate(eng) if v >= recog_args.auto_vad_threshold]
                    else:
                        vad_idx = None
                    ae_out, latent_out = pm_model(inp, torch.IntTensor([inp.shape[1]]))
                    pm_scores[idx], llhood = pm_model.scoring_fn(inp, ae_out,
                                                                 latent_out,
                                                                 temp=recog_args.temperature,
                                                                 type=recog_args.score_type,
                                                                 chop=recog_args.chop,
                                                                 out_dist=recog_args.vae_output_distribution,
                                                                 vad_idx=vad_idx)
                    if pm_type == "vae_old":
                        llhood = torch.from_numpy(llhood)
                all_llhoods.append(llhood)
        elif pm_type == "weighted":
            pm_scores = torch.as_tensor(pm_models)
        elif pm_type == "average":
            # pm_scores = torch.ones((x.size(0), self.stream_num)) / float(self.stream_num)
            recog_args.average_ctc = True
            recog_args.average_attn = True
            pm_scores = torch.ones(self.stream_num) / float(self.stream_num)
        elif pm_type == "mmeasure" or pm_type == "mmeasure_addlm":
            pm_scores_ctc = torch.zeros(self.stream_num)
            pm_scores_attn = torch.zeros(self.stream_num)

        if recog_args.weight_combination == "per_frame_posteriors":
            # def per_frame_post_ctc(logits):
            #    p_x_y_app = torch.exp(logits.logsumexp(-1, keepdim=True))
            #    return torch.log((F.softmax(logits, dim=-1) * p_x_y_app/p_x_y_app.sum(0)).sum(0).squeeze(0))
            # def per_frame_post_attn(logits, i):
            #    p_x_y_app = torch.exp(logits.logsumexp(-1, keepdim=True))
            #    return torch.log((F.softmax(logits, dim=-1) * p_x_y_app/p_x_y_app.sum(0)).sum(0))
            pm_sc_f_ctc = lambda logits: F.log_softmax(logits.logsumexp(dim=0) - np.log(logits.size(0)),
                                                       dim=-1).squeeze(0)
            pm_sc_f_attn = lambda logits, i: F.log_softmax(logits.logsumexp(dim=0) - np.log(logits.size(0)), dim=-1)
        elif recog_args.weight_combination == "per_frame_logits":
            pm_sc_f_ctc = lambda logits: F.log_softmax(logits.mean(0), dim=-1).squeeze(0)
            pm_sc_f_attn = lambda logits, i: F.log_softmax(logits.mean(0), dim=-1)  # "Cannot calculate from posterios"
        elif pm_type != "average":
            if recog_args.weight_combination.startswith("per_frame"):
                pm_scores = sm(
                    reduce_len_per_frame(torch.stack(all_llhoods)))  # TODO: should we first compute sm and then reduce?
                pm_sc_f_attn = lambda logits, i: 1 / 0  # TBD #lambda posteriors, i : (posteriors * pm_scores[:, i].unsqueeze(-1)).mean(0)
            else:
                if recog_args.weight_combination == "alternative":
                    pm_scores = torch.mean(sm(torch.stack(all_llhoods)), dim=-1).unsqueeze(-1)
                elif recog_args.weight_combination == "standard":
                    pm_scores = sm(pm_scores).unsqueeze(-1)

                pm_sc_f_attn = lambda logits, _: torch.log((F.softmax(logits, dim=-1) * pm_scores.unsqueeze(-1)).sum(0))
            pm_sc_f_ctc = lambda logits: torch.log(
                (F.softmax(logits, dim=-1).squeeze(1) * pm_scores.unsqueeze(-1)).sum(0))

        pm_sc_f_ctc = (lambda logits: torch.log(
            (F.softmax(logits, dim=-1)).squeeze(1).mean(0))) if recog_args.average_ctc else pm_sc_f_ctc
        pm_sc_f_attn = (
            lambda logits, _: torch.log(F.softmax(logits, dim=-1).mean(0))) if recog_args.average_attn else pm_sc_f_attn

        return all_llhoods, pm_scores, pm_sc_f_ctc, pm_sc_f_attn

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        logging.info("Using score type {:s}".format(recog_args.score_type))
        logging.info("Using temperature {:f}".format(recog_args.temperature))
        if recog_args.general_p:
            logging.info(
                "Using generalized mean classifier combination with parameter p={:f}".format(recog_args.general_p))
        else:
            logging.info("Using {:s} rule of classifier combination".format(recog_args.rule))

        all_enc_outputs = []
        for model in self.all_models:
            all_enc_outputs.append(model.encode(x).unsqueeze(0))
        all_enc_outputs = torch.stack(all_enc_outputs)
        print("SH: ", all_enc_outputs.shape, x.shape)
        # Get performance monitor Scores
        if recog_args.oracle_wts:
            oracle_run = True
        else:
            oracle_run = False
        if oracle_run:
            pm_scores_ctc = self.all_pms['lpz']
            pm_scores_attn = self.all_pms['attn']
            pm_scores_nn = pm_scores_attn
        else:
            if "joint" in recog_args.pm_type:
                #TODO:
                exit("Joint is not supported in this version")
            elif "enc" in recog_args.pm_type:
                def vae_inp_f(x, enc_x, idx):
                    if recog_args.enc_feat_transform_type == 'pca':
                        return self.apply_pca(enc_x[idx], self.all_enc_transform[idx])
                    elif recog_args.enc_feat_transform_type == 'cmvn':
                        return self.apply_cmvn(enc_x[idx], self.all_enc_transform[idx][0], self.all_enc_transform[idx][1])
                vae_type = recog_args.pm_type.replace("enc_", "").replace("_enc", "")
                vae_type = "vae_old" if vae_type == "vae" else vae_type
                all_llhoods, pm_scores, pm_sc_f_ctc, pm_sc_f_attn = self.get_sc_f(x, all_enc_outputs, vae_type, self.all_enc_pms, vae_inp_f, recog_args)
            else:
                vae_inp_f = lambda x, enc_x, idx: torch.as_tensor(x).unsqueeze(0)
                all_llhoods, pm_scores, pm_sc_f_ctc, pm_sc_f_attn = self.get_sc_f(x, all_enc_outputs, recog_args.pm_type, self.all_pms, vae_inp_f, recog_args)

        if oracle_run:
            logging.info("Using oracle performance monitor")
        elif recog_args.pm_type == "mmeasure" or recog_args.pm_type == "mmeasure_addlm":
            logging.info("Using {:s} performance monitor".format(recog_args.pm_type))
        else:
            pm_scores_nn = deepcopy(pm_scores.data.numpy())
            for idx, psc in enumerate(pm_scores):
                logging.info("Confidence score for ASR " + str(idx) + " is " + str(psc.mean(-1).detach().numpy()))
                if len(all_llhoods) > idx:
                    logging.info("Llhood for ASR " + str(idx) + " is " + str(torch.Tensor(all_llhoods[idx]).mean()))

        if self.all_models[0].mtlalpha == 1.0:
            # Assuming all models are pure CTC
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.all_models[0].mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            enc_ctc = torch.stack([model.ctc.ctc_lo(x) for model, x in zip(self.all_models, all_enc_outputs)])
            lpz = pm_sc_f_ctc(enc_ctc).unsqueeze(1)
            lpz = torch.argmax(lpz, dim=2)


            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.all_models[0].blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.all_models[0].sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return recog_args.temperature, None, None, nbest_hyps
        elif self.all_models[0].mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            #all_lpz = []

            #for idx, model in enumerate(self.all_models):
            #    all_lpz.append(model.ctc.log_softmax(all_enc_outputs[idx]).squeeze(0))
            #    if not oracle_run:
            #        if recog_args.pm_type == "mmeasure" or recog_args.pm_type == "mmeasure_addlm":
            #            pm_scores_ctc[idx] = torch.exp(
            #                recog_args.temperature * self.mmeasure_loss(torch.exp(all_lpz[idx])))
            #if not oracle_run:
            #    if recog_args.pm_type == "mmeasure" or recog_args.pm_type == "mmeasure_addlm":
            #        pm_scores_nn = deepcopy(np.log(pm_scores_ctc.data.numpy()) / recog_args.temperature)
            #        pm_scores_ctc = pm_scores_ctc / torch.sum(pm_scores_ctc)
            #        pm_scores = pm_scores_ctc
            #        for idx, psc in enumerate(pm_scores_ctc):
            #            logging.info("CTC Confidence score for ASR " + str(idx) + " is " + str(psc.detach().numpy()))
            if oracle_run:
                pm_scores = pm_scores_ctc
                for idx, psc in enumerate(pm_scores_ctc):
                    logging.info("Oracle CTC Confidence score for ASR " + str(idx) + " is " + str(psc))
                for idx, psc in enumerate(pm_scores_attn):
                    logging.info("Oracle ATTN Confidence score for ASR " + str(idx) + " is " + str(psc))


            enc_ctc = torch.stack([model.ctc.ctc_lo(x) for model, x in zip(self.all_models, all_enc_outputs)])
            #TODO: here lpz is log?
            lpz = pm_sc_f_ctc(enc_ctc)
            print("LPZ",enc_ctc.shape, lpz.shape)




            #lpz = 0 * all_lpz[0]
            #for idx, one_lpz in enumerate(all_lpz):
            #    print("TT", one_lpz.shape, pm_scores[idx].shape)
            #    if recog_args.general_p:
            #        lpz += torch.pow(torch.exp(one_lpz), recog_args.general_p) * pm_scores[idx]
            #    elif recog_args.rule == 'product':
            #        lpz += one_lpz * pm_scores[idx].unsqueeze(-1)
            #    elif recog_args.rule == 'sum':
            #        lpz += torch.exp(one_lpz) * pm_scores[idx].unsqueeze(-1)
            #    else:
            #        logging.error("rule can only be of types 'sum' or 'product'")
            #if recog_args.general_p:
            #    lpz = torch.log(lpz) / recog_args.general_p
            #elif recog_args.rule == "sum":
            #        lpz = torch.log(lpz)
        else:
            lpz = None

        h = all_enc_outputs[0].squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.all_models[0].sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            #print("SHHH", lpz.detach().numpy().shape)
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.all_models[0].eos, np)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        all_compiled_local_att_scores = []
        for i in range(self.stream_num):
            all_compiled_local_att_scores.append(np.zeros((maxlen, lpz.shape[1])))
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            best_score = -np.infty
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                all_local_att_scores = []
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.all_models[0].decoder.forward_one_step, (ys, ys_mask, all_enc_outputs[0])
                        )
                    for enc_output in all_enc_outputs:
                        local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                        all_local_att_scores.append(local_att_scores)
                else:
                    # Keep only attention values from the best hypothesis
                    if hyp["score"] > best_score:
                        best_score = hyp["score"]
                        update = True
                    else:
                        update = False

                    if not recog_args.ignore_attn: # This is just speedup, so we don't need to use ATTN-based decoder
                        #all_local_att_scores = torch.stack([model.decoder.forward_one_step(ys, ys_mask, x)[0] for model, x in zip(self.all_models, all_enc_outputs)])
                        #print(type(model.decoder))
                        all_local_att_scores = torch.stack(
                            [super_ugly_hack_how_to_get_logits_from_one_specific_model(model.decoder, ys, ys_mask, x) for model, x in
                             zip(self.all_models, all_enc_outputs)])
                        # TODO put there logits instead of posteriors and fix i-problem
                        local_att_scores = pm_sc_f_attn(all_local_att_scores, i)

                    # TODO: ignoring update variable!

                    #for idx, enc_output in enumerate(all_enc_outputs):
                    #    local_att_scores = self.all_models[idx].decoder.forward_one_step(ys, ys_mask, enc_output)[0]
                    #    if update and not oracle_run:
                    #        if recog_args.pm_type == "mmeasure_addlm":
                    #            all_compiled_local_att_scores[idx][i, :] = local_att_scores[0,:].detach().numpy() + recog_args.lm_weight * local_lm_scores[0, :].detach()
                    #        else:
                    #            all_compiled_local_att_scores[idx][i, :] = local_att_scores[0, :].detach()
                    #    all_local_att_scores.append(local_att_scores)
                        #if not oracle_run:
                        #    if recog_args.pm_type == "mmeasure" or recog_args.pm_type == "mmeasure_addlm":
                        #        if i >= 10:
                        #            pm_scores_attn[idx] = torch.exp(recog_args.temperature * self.mmeasure_loss(
                        #                torch.exp(torch.from_numpy(all_compiled_local_att_scores[idx][0:i - 1, :]))))
                        #        else:
                        #            pm_scores_attn = pm_scores_ctc

                #if oracle_run:
                #    pm_scores = pm_scores_attn
                #elif recog_args.pm_type == "mmeasure" or recog_args.pm_type == "mmeasure_addlm":
                #    pm_scores_nn = deepcopy(np.log(pm_scores_attn.data.numpy()) / recog_args.temperature)
                #    pm_scores_attn = pm_scores_attn / torch.sum(pm_scores_attn)
                #    pm_scores = pm_scores_attn
                #    log = "Prediction" + str(i) + "-> ATTN Confidence score for "
                #    for idx, psc in enumerate(pm_scores_attn):
                #        log = log + " ASR " + str(idx) + ": " + str(psc.detach().numpy())
                #    logging.info(log)



                #local_att_scores = 0 * all_local_att_scores[0]
                #for idx, one_attn_score in enumerate(all_local_att_scores):
                    #pm_score = pm_scores[idx][i] if recog_args.weight_combination.startswith("per_frame") else pm_scores[idx]
                    #if recog_args.general_p:
                    #    local_att_scores += torch.pow(torch.exp(one_attn_score), recog_args.general_p) * pm_scores[
                    #        idx]
                    #elif recog_args.rule == 'product':
                    #    local_att_scores += one_attn_score * pm_score
                    #elif recog_args.rule == 'sum':
                    #    local_att_scores += torch.exp(one_attn_score) * pm_score
                    #else:
                    #    logging.error("rule can only be of types 'sum' or 'product'")
                #if recog_args.general_p:
                #    local_att_scores = torch.log(local_att_scores) / recog_args.general_p
                #elif recog_args.rule == "sum":
                #    local_att_scores = torch.log(local_att_scores)


                if recog_args.ignore_attn:
                    ctc_scores, ctc_states = ctc_prefix_score(hyp["yseq"], np.arange(lpz.shape[-1]), hyp["ctc_state_prev"])
                    local_scores = ctc_weight * torch.from_numpy(ctc_scores - hyp["ctc_score_prev"]) + recog_args.lm_weight * local_lm_scores
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)
                    joint_best_ids = local_best_ids # TODO: I don't know if this is legit (I guess joint_best_ids shouldn't be used at all)
                else:
                    if rnnlm:
                        local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                    else:
                        local_scores = local_att_scores

                    if lpz is not None:
                        local_best_scores, local_best_ids = torch.topk(
                            local_att_scores, ctc_beam, dim=1
                        )
                        ctc_scores, ctc_states = ctc_prefix_score(hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"])
                        local_scores = (1.0 - ctc_weight) * local_att_scores[
                                                            :, local_best_ids[0]
                                                            ] + ctc_weight * torch.from_numpy(
                            ctc_scores - hyp["ctc_score_prev"]
                        )
                        if rnnlm:
                            local_scores += (
                                    recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                            )
                        local_best_scores, joint_best_ids = torch.topk(
                            local_scores, beam, dim=1
                        )
                        local_best_ids = local_best_ids[:, joint_best_ids[0]]
                    else:
                        local_best_scores, local_best_ids = torch.topk(
                            local_scores, beam, dim=1
                        )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.all_models[0].eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.all_models[0].eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
                     : min(len(ended_hyps), recog_args.nbest)
                     ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        #for idx, x in enumerate(all_lpz):
        #    all_lpz[idx] = all_lpz[idx].detach().numpy()

        # Update the temperature
        if recog_args.update_temp:
            recog_args.temperature = self.update_temp(recog_args.temperature, pm_scores_nn, all_llhoods,
                                                      all_compiled_local_att_scores, lr=10)

        comp = {}
        #comp["lpz"] = all_lpz
        comp["attn"] = all_compiled_local_att_scores
        comp["all_llhoods"] = all_llhoods
        return recog_args.temperature, pm_scores_nn, comp, nbest_hyps

    def recognize_joint(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech with joint PM.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        logging.info("Using score type {:s}".format(recog_args.score_type))
        logging.info("Using temperature {:f}".format(recog_args.temperature))
        if recog_args.general_p:
            logging.info(
                "Using generalized mean classifier combination with parameter p={:f}".format(recog_args.general_p))
        else:
            logging.info("Using {:s} rule of classifier combination".format(recog_args.rule))

        all_enc_outputs = []
        for model in self.all_models:
            all_enc_outputs.append(model.encode(x).unsqueeze(0))

        all_llhoods = []
        all_norm_enc_out = []
        all_enc_outputs_np = []
        for enn in all_enc_outputs:
            all_enc_outputs_np.append(enn[0].data.numpy())

        # Get performance monitor Scores
        pm_scores = torch.zeros(self.stream_num)
        if recog_args.pm_type == "joint_vae":
            pm_scores_inp = torch.zeros(self.stream_num)
            pm_scores_enc = torch.zeros(self.stream_num)

            # External Model
            for idx, pm_model in enumerate(self.all_pms):
                ae_out, latent_out = pm_model(torch.as_tensor(x).unsqueeze(0), torch.IntTensor([x.shape[0]]))
                pm_scores_inp[idx], llhood = pm_model.vae_loss_score(torch.as_tensor(x).unsqueeze(0), ae_out,
                                                                     latent_out,
                                                                     out_dist='laplace', temp=recog_args.temperature,
                                                                     type=recog_args.score_type, chop=recog_args.chop)
                all_llhoods.append(llhood)

            # Internal Model
            for idx, pm_model in enumerate(self.all_enc_pms):
                if recog_args.enc_feat_transform_type == 'pca':
                    enc_out_after_transform = self.apply_pca(all_enc_outputs[idx], self.all_enc_transform[idx])
                elif recog_args.enc_feat_transform_type == 'cmvn':
                    enc_out_after_transform = self.apply_cmvn(all_enc_outputs[idx], self.all_enc_transform[idx][0],
                                                              self.all_enc_transform[idx][1])

                ae_out, latent_out = pm_model(enc_out_after_transform,
                                              torch.IntTensor([enc_out_after_transform.shape[1]]))
                pm_scores_enc[idx], llhood = pm_model.vae_loss_score(enc_out_after_transform, ae_out, latent_out,
                                                                     out_dist=recog_args.vae_output_distribution,
                                                                     temp=recog_args.temperature,
                                                                     type=recog_args.score_type, chop=recog_args.chop)

            # Combine scores
            for idx, x in enumerate(pm_scores_inp):
                pm_scores[idx] = torch.exp(torch.log(pm_scores_enc[idx]) + torch.log(pm_scores_inp[idx]))

            pm_scores_nn = deepcopy(pm_scores.data.numpy())
            pm_scores_nn_inp = deepcopy(pm_scores_inp.data.numpy())
            pm_scores_nn_enc = deepcopy(pm_scores_enc.data.numpy())
            pm_scores = pm_scores / torch.sum(pm_scores)
            for idx, psc in enumerate(pm_scores):
                temp_inp = (pm_scores_inp / torch.sum(pm_scores_inp)).detach().numpy()
                temp_enc = (pm_scores_enc / torch.sum(pm_scores_enc)).detach().numpy()
                logging.info("Confidence score for ASR {:d} (final/input/encoder) = {:f}/{:f}/{:f}".format(idx,
                                                                                                           psc.detach().numpy(),
                                                                                                           temp_inp[
                                                                                                               idx],
                                                                                                           temp_enc[
                                                                                                               idx]))
        elif recog_args.pm_type == "enc_vae" or recog_args.pm_type == "enc_vae_ff":
            # Internal Model only
            sm = torch.nn.Softmax(dim=0)
            for idx, pm_model in enumerate(self.all_enc_pms):
                if recog_args.enc_feat_transform_type == 'pca':
                    enc_out_after_transform = self.apply_pca(all_enc_outputs[idx], self.all_enc_transform[idx])
                elif recog_args.enc_feat_transform_type == 'cmvn':
                    enc_out_after_transform = self.apply_cmvn(all_enc_outputs[idx], self.all_enc_transform[idx][0],
                                                              self.all_enc_transform[idx][1])
                all_norm_enc_out.append(enc_out_after_transform[0].data.numpy())

                if recog_args.pm_type == "enc_vae":
                    ae_out, latent_out = pm_model(enc_out_after_transform,
                                                  torch.IntTensor([enc_out_after_transform.shape[1]]))
                    if recog_args.auto_vad:
                        eng = np.mean(enc_out_after_transform[0].data.numpy() ** 2, axis=1)
                        vad_idx = [idx for idx, v in enumerate(eng) if v >= recog_args.auto_vad_threshold]
                    pm_scores[idx], llhood = pm_model.vae_loss_score(enc_out_after_transform, ae_out, latent_out,
                                                                     out_dist=recog_args.vae_output_distribution,
                                                                     temp=recog_args.temperature,
                                                                     type=recog_args.score_type, chop=recog_args.chop,
                                                                     vad_idx=vad_idx)
                elif recog_args.pm_type == "enc_vae_ff":
                    pm_scores[idx], llhood = pm_model.vae_loss_score(torch.squeeze(enc_out_after_transform, 0))

                all_llhoods.append(llhood)

            pm_scores_nn = deepcopy(pm_scores.data.numpy())
            if recog_args.weight_combination == "alternative":
                pm_scores = torch.mean(sm(torch.Tensor(all_llhoods)), dim=-1)
            else:  # recog_args.weight_combination == "standard"
                pm_scores = sm(pm_scores)

            for idx, psc in enumerate(pm_scores):
                logging.info("Confidence score for ASR (final) " + str(idx) + " is " + str(
                    psc.detach().numpy()) + " and log-likelihood value is " + str(pm_scores_nn[idx]))

        if self.all_models[0].mtlalpha == 1.0:
            # Assuming all models are pure CTC
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.all_models[0].mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            all_lpz = []
            for idx, model in enumerate(self.all_models):
                all_lpz.append(model.ctc.softmax(all_enc_outputs[idx]))

            lpz = 0 * all_lpz[0]
            for idx, pm_model in enumerate(self.all_pms):
                lpz += pm_scores[idx] * all_lpz[idx]
            lpz = torch.argmax(lpz, dim=2)

            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.all_models[0].mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            all_lpz = []
            for idx, model in enumerate(self.all_models):
                all_lpz.append(model.ctc.log_softmax(all_enc_outputs[idx]).squeeze(0))
            lpz = 0 * all_lpz[0]
            for idx, one_lpz in enumerate(all_lpz):
                if recog_args.general_p:
                    lpz += torch.pow(torch.exp(one_lpz), recog_args.general_p) * pm_scores[idx]
                elif recog_args.rule == 'product':
                    lpz += one_lpz * pm_scores[idx]
                elif recog_args.rule == 'sum':
                    lpz += torch.exp(one_lpz) * pm_scores[idx]
                else:
                    logging.error("rule can only be of types 'sum' or 'product'")
            if recog_args.general_p:
                lpz = torch.log(lpz) / recog_args.general_p
            elif recog_args.rule == "sum":
                lpz = torch.log(lpz)
        else:
            lpz = None

        h = all_enc_outputs[0].squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.all_models[0].sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.all_models[0].eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        all_compiled_local_att_scores = []
        for i in range(self.stream_num):
            all_compiled_local_att_scores.append(np.zeros((maxlen, lpz.shape[1])))
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            best_score = -np.infty
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                all_local_att_scores = []
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.all_models[0].decoder.forward_one_step, (ys, ys_mask, all_enc_outputs[0])
                        )
                    for enc_output in all_enc_outputs:
                        local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                        all_local_att_scores.append(local_att_scores)
                else:
                    # Keep only attention values from the best hypothesis
                    if hyp["score"] > best_score:
                        best_score = hyp["score"]
                        update = True
                    else:
                        update = False
                    for idx, enc_output in enumerate(all_enc_outputs):
                        local_att_scores = self.all_models[idx].decoder.forward_one_step(
                            ys, ys_mask, enc_output
                        )[0]
                        if update:
                            all_compiled_local_att_scores[idx][i, :] = local_att_scores[0, :].detach().numpy()
                        all_local_att_scores.append(local_att_scores)

                local_att_scores = 0 * all_local_att_scores[0]
                for idx, one_attn_score in enumerate(all_local_att_scores):
                    if recog_args.general_p:
                        local_att_scores += torch.pow(torch.exp(one_attn_score), recog_args.general_p) * pm_scores[
                            idx]
                    elif recog_args.rule == 'product':
                        local_att_scores += one_attn_score * pm_scores[idx]
                    elif recog_args.rule == 'sum':
                        local_att_scores += torch.exp(one_attn_score) * pm_scores[idx]
                    else:
                        logging.error("rule can only be of types 'sum' or 'product'")
                if recog_args.general_p:
                    local_att_scores = torch.log(local_att_scores) / recog_args.general_p
                elif recog_args.rule == "sum":
                    local_att_scores = torch.log(local_att_scores)

                if rnnlm:
                    local_scores = (
                            local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                                                        :, local_best_ids[0]
                                                        ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                                recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.all_models[0].eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.all_models[0].eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
                     : min(len(ended_hyps), recog_args.nbest)
                     ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        for idx, _ in enumerate(all_lpz):
            all_lpz[idx] = all_lpz[idx].detach().numpy()

        for idx, _ in enumerate(all_enc_outputs):
            all_enc_outputs[idx] = all_enc_outputs[idx].detach().numpy()

        # Update the temperature
        if recog_args.update_temp:
            recog_args.temperature = self.update_temp(recog_args.temperature, pm_scores_nn, all_llhoods,
                                                      all_compiled_local_att_scores, lr=10)

        comp = {}
        comp["lpz"] = all_lpz
        comp["attn"] = all_compiled_local_att_scores
        comp['enc_outputs'] = all_norm_enc_out
        comp['all_llhoods'] = all_llhoods
        if recog_args.pm_type == "joint_vae":
            pm_scores_nn = {'inp': pm_scores_nn_inp, 'enc': pm_scores_nn_enc}
        return recog_args.temperature, pm_scores_nn, comp, nbest_hyps

    def recognize_self_confidence(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech with self confidence.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        logging.info("Using score type {:s}".format(recog_args.score_type))
        logging.info("Using temperature {:f}".format(recog_args.temperature))
        if recog_args.general_p:
            logging.info(
                "Using generalized mean classifier combination with parameter p={:f}".format(recog_args.general_p))
        else:
            logging.info("Using {:s} rule of classifier combination".format(recog_args.rule))

        all_enc_outputs = []
        for model in self.all_models:
            all_enc_outputs.append(model.encode(x).unsqueeze(0))

        all_llhoods = []

        # Get performance monitor Scores
        pm_scores = torch.zeros(self.stream_num)
        if recog_args.pm_type == "joint_vae":
            pm_scores_inp = torch.zeros(self.stream_num)
            pm_scores_enc = torch.zeros(self.stream_num)

            # External Model
            for idx, pm_model in enumerate(self.all_pms):
                ae_out, latent_out = pm_model(torch.as_tensor(x).unsqueeze(0), torch.IntTensor([x.shape[0]]))
                pm_scores_inp[idx], llhood = self.vae_loss(torch.as_tensor(x).unsqueeze(0), ae_out, latent_out,
                                                           out_dist='laplace', temp=recog_args.temperature,
                                                           type=recog_args.score_type, chop=recog_args.chop)
                all_llhoods.append(llhood)

            # Internal Model
            for idx, pm_model in enumerate(self.all_enc_pms):
                enc_out_after_cmvn = self.apply_cmvn(all_enc_outputs[idx], self.all_enc_cmvn[idx][0],
                                                     self.all_enc_cmvn[idx][1])
                ae_out, latent_out = pm_model(enc_out_after_cmvn, torch.IntTensor([enc_out_after_cmvn.shape[1]]))
                pm_scores_enc[idx], llhood = self.vae_loss(enc_out_after_cmvn, ae_out, latent_out,
                                                           out_dist='laplace', temp=recog_args.temperature,
                                                           type=recog_args.score_type, chop=recog_args.chop)
            # Combine scores
            for idx, x in enumerate(pm_scores_inp):
                pm_scores[idx] = torch.exp(torch.log(pm_scores_enc[idx]) + torch.log(pm_scores_inp[idx]))

            pm_scores_nn = deepcopy(pm_scores.data.numpy())
            pm_scores_nn_inp = deepcopy(pm_scores_inp.data.numpy())
            pm_scores_nn_enc = deepcopy(pm_scores_enc.data.numpy())
            pm_scores = pm_scores / torch.sum(pm_scores)
            for idx, psc in enumerate(pm_scores):
                temp_inp = (pm_scores_inp / torch.sum(pm_scores_inp)).detach().numpy()
                temp_enc = (pm_scores_enc / torch.sum(pm_scores_enc)).detach().numpy()
                logging.info("Confidence score for ASR {:d} (final/input/encoder) = {:f}/{:f}/{:f}".format(idx,
                                                                                                           psc.detach().numpy(),
                                                                                                           temp_inp[
                                                                                                               idx],
                                                                                                           temp_enc[
                                                                                                               idx]))
        elif recog_args.pm_type == "enc_vae":

            # Internal Model only
            for idx, pm_model in enumerate(self.all_enc_pms):
                enc_out_after_cmvn = self.apply_cmvn(all_enc_outputs[idx], self.all_enc_cmvn[idx][0],
                                                     self.all_enc_cmvn[idx][1])
                ae_out, latent_out = pm_model(enc_out_after_cmvn, torch.IntTensor([enc_out_after_cmvn.shape[1]]))
                pm_scores[idx], llhood = self.vae_loss(enc_out_after_cmvn, ae_out, latent_out,
                                                       out_dist='laplace', temp=recog_args.temperature,
                                                       type=recog_args.score_type, chop=recog_args.chop)

            pm_scores_nn = deepcopy(pm_scores.data.numpy())
            pm_scores = pm_scores / torch.sum(pm_scores)
            for idx, psc in enumerate(pm_scores):
                logging.info("Confidence score for ASR (final) " + str(idx) + " is " + str(psc.detach().numpy()))

        if self.all_models[0].mtlalpha == 1.0:
            # Assuming all models are pure CTC
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.all_models[0].mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            all_lpz = []
            for idx, model in enumerate(self.all_models):
                all_lpz.append(model.ctc.softmax(all_enc_outputs[idx]))

            lpz = 0 * all_lpz[0]
            for idx, pm_model in enumerate(self.all_pms):
                lpz += pm_scores[idx] * all_lpz[idx]
            lpz = torch.argmax(lpz, dim=2)

            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.all_models[0].mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            all_lpz = []
            for idx, model in enumerate(self.all_models):
                all_lpz.append(model.ctc.log_softmax(all_enc_outputs[idx]).squeeze(0))
            lpz = 0 * all_lpz[0]
            for idx, one_lpz in enumerate(all_lpz):
                if recog_args.general_p:
                    lpz += torch.pow(torch.exp(one_lpz), recog_args.general_p) * pm_scores[idx]
                elif recog_args.rule == 'product':
                    lpz += one_lpz * pm_scores[idx]
                elif recog_args.rule == 'sum':
                    lpz += torch.exp(one_lpz) * pm_scores[idx]
                else:
                    logging.error("rule can only be of types 'sum' or 'product'")
            if recog_args.general_p:
                lpz = torch.log(lpz) / recog_args.general_p
            elif recog_args.rule == "sum":
                lpz = torch.log(lpz)
        else:
            lpz = None

        h = all_enc_outputs[0].squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.all_models[0].sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.all_models[0].eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        all_compiled_local_att_scores = []
        for i in range(self.stream_num):
            all_compiled_local_att_scores.append(np.zeros((maxlen, lpz.shape[1])))
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            best_score = -np.infty
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                all_local_att_scores = []
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.all_models[0].decoder.forward_one_step, (ys, ys_mask, all_enc_outputs[0])
                        )
                    for enc_output in all_enc_outputs:
                        local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                        all_local_att_scores.append(local_att_scores)
                else:
                    # Keep only attention values from the best hypothesis
                    if hyp["score"] > best_score:
                        best_score = hyp["score"]
                        update = True
                    else:
                        update = False
                    for idx, enc_output in enumerate(all_enc_outputs):
                        local_att_scores = self.all_models[idx].decoder.forward_one_step(
                            ys, ys_mask, enc_output
                        )[0]
                        if update:
                            all_compiled_local_att_scores[idx][i, :] = local_att_scores[0, :].detach().numpy()
                        all_local_att_scores.append(local_att_scores)

                local_att_scores = 0 * all_local_att_scores[0]
                for idx, one_attn_score in enumerate(all_local_att_scores):
                    if recog_args.general_p:
                        local_att_scores += torch.pow(torch.exp(one_attn_score), recog_args.general_p) * pm_scores[
                            idx]
                    elif recog_args.rule == 'product':
                        local_att_scores += one_attn_score * pm_scores[idx]
                    elif recog_args.rule == 'sum':
                        local_att_scores += torch.exp(one_attn_score) * pm_scores[idx]
                    else:
                        logging.error("rule can only be of types 'sum' or 'product'")
                if recog_args.general_p:
                    local_att_scores = torch.log(local_att_scores) / recog_args.general_p
                elif recog_args.rule == "sum":
                    local_att_scores = torch.log(local_att_scores)

                if rnnlm:
                    local_scores = (
                            local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                                                        :, local_best_ids[0]
                                                        ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                                recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.all_models[0].eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.all_models[0].eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
                     : min(len(ended_hyps), recog_args.nbest)
                     ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        for idx, x in enumerate(all_lpz):
            all_lpz[idx] = all_lpz[idx].detach().numpy()

        # Update the temperature
        if recog_args.update_temp:
            recog_args.temperature = self.update_temp(recog_args.temperature, pm_scores_nn, all_llhoods,
                                                      all_compiled_local_att_scores, lr=10)

        comp = {}
        comp["lpz"] = all_lpz
        comp["attn"] = all_compiled_local_att_scores
        if recog_args.pm_type == "joint_vae":
            pm_scores_nn = {'inp': pm_scores_nn_inp, 'enc': pm_scores_nn_enc}
        return recog_args.temperature, pm_scores_nn, comp, nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                    isinstance(m, MultiHeadedAttention)
                    or isinstance(m, DynamicConvolution)
                    or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
