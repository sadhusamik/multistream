# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import copy
import json
import logging
import math
import os
import sys

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.nn.parallel import data_parallel

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import format_mulenc_args
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import freeze_modules
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib

matplotlib.use("Agg")

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


def encoder_feats(args):
    """Encode the input features with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model, training=False)
    assert isinstance(model, ASRInterface)

    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    all_encoded_feats = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info("(%d/%d) encoding " + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)
            feat = (
                feat[0][0]
                if args.num_encs == 1
                else [feat[idx][0] for idx in range(model.num_encs)]
            )
            encoded_feats = model.encode(feat)
            all_encoded_feats[name] = encoded_feats

    return all_encoded_feats


def recog_cl(args):
    """Decode with several parallel models with the given args.

    Args:
        args (namespace): The program arguments.

    """
    from src.espnet_extensions.nets.e2e_asr_transformer_cl import E2E
    from src.nnet.models.VAE import nnetVAE, VAE_LSTM, FF_VAE, FF_VAE_Rotation

    set_deterministic_pytorch(args)
    model_list = args.model_list.strip().split(',')
    all_models = []
    all_train_args = []
    for i in model_list:
        print(i)
        # asr_model, train_args = load_trained_model(i, training=False)
        asr_model, train_args = load_trained_model(i)
        asr_model.eval()
        assert isinstance(asr_model, ASRInterface)
        asr_model.recog_args = args
        all_models.append(asr_model)
        all_train_args.append(train_args)

    if args.pm_type is not "weighted":
        pm_list = args.pm_list.strip().split(',')
    all_pm_models = []

    if args.average_ctc:
        logging.info("Using MEAN for combining CTCs!")
    if args.ignore_attn:
        logging.info("Ignoring score from attention and using CTC only!")
    elif args.average_attn:
        logging.info("Using MEAN for combining attention!")

    if args.weight_combination == "standard":
        logging.info("Using STANDARD (first reduction in time, then softmax) way of computing score")
    elif args.weight_combination == "alternative":
        logging.info("Using ALTERNATIVE (first softmax per frame - over models - then mean in time) way of computing score")
    elif args.weight_combination == "per_frame":
        logging.info("Using PER FRAME combination way of computing score")
    elif args.weight_combination == "per_frame_logits":
        logging.info("Using PER FRAME weight derived from LOGITS to combine models")
    elif args.weight_combination == "per_frame_posteriors":
        logging.info("Using PER FRAME weight derived from POSTERIORS to compute score")
    else:
        logging.error("Wrong option for computing score (weight_combination).")
        raise ValueError("Wrong option for computing score (weight_combination), Use 'alternative', 'standard' 'per_frame', 'per_frame_logits' or 'per_frame_posteriors'.")

    if args.oracle_wts:
        logging.info("Using oracle weights")
    else:
        if args.pm_type == "vae":
            logging.info("Using variational auto-encoder performance monitor")
        elif args.pm_type == "vae_old":
            logging.info("Using variational auto-encoder performance monitor (old version)")
        elif args.pm_type == "vae_ff":
            logging.info("Using variational auto-encoder performance monitor with FF NN")
        elif args.pm_type == "average":
            logging.info("Averaging models (using weight equal to 1/num_of_models)")
        elif args.pm_type == "weighted":
            if args.pm_weights:
                logging.info("Using combination weights provided as inputs")
            else:
                logging.error("You need to provide pm_weights if pm_type is 'weighted")
                raise ValueError("You need to provide pm_weights if pm_type is 'weighted")
        elif args.pm_type == "mmeasure" or args.pm_type == "mmeasure_addlm":
            logging.info("Using mmeasure performance monitor")
        else:
            raise ValueError("Performance monitor can only be 'mmeasure', 'ae' or 'vae', 'mmeasure_addlm")

    if args.pm_save_path:
        all_pm_scores = {}
    if args.posterior_save_path:
        all_lpz_vals = {}

    import pickle
    if args.oracle_wts:
        all_oracle_wts = pickle.load(open(args.oracle_wts, 'rb'))

    device = "cpu"
    if args.oracle_wts:
        all_pm_models = None
    else:
        if args.pm_type == "vae" or args.pm_type == "ae" or args.pm_type == "vae_ff" or args.pm_type == "vae_old":
            for i in pm_list:
                model_dict = torch.load(i, map_location=lambda storage, loc: storage)
                if args.pm_type == "vae_ff":
                    config = model_dict["config"]
                    if config.rotation_vae:
                        pm_model = FF_VAE_Rotation(config, device)
                    else:
                        pm_model = FF_VAE(config, device)
                elif args.pm_type == "vae":
                    print(model_dict.keys())
                    config = model_dict["conf"]
                    pm_model = VAE_LSTM(config, use_gpu=False)
                elif args.pm_type == "vae_old":
                    pm_model = nnetVAE(model_dict["feature_dim"], model_dict["encoder_num_layers"],
                                       model_dict["decoder_num_layers"], model_dict["hidden_dim"], model_dict["bn_dim"],
                                       dropout=0, use_gpu=False)
                pm_model.load_state_dict(model_dict["model_state_dict"])
                pm_model.eval()
                all_pm_models.append(pm_model)
        elif args.pm_type == "weighted":
            all_pm_models = args.pm_weights.strip().split(',')
            all_pm_models = [float(x) for x in all_pm_models]
        else:
            all_pm_models = None

    model = E2E(all_models, all_pm_models)

    if args.streaming_mode and "transformer" in all_train_args[0].model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")

    for idx, i in enumerate(all_models):
        logging.info(
            " Total parameter of the ASR model number " + str(idx + 1) + " = "
            + str(sum(p.numel() for p in i.parameters()))
        )

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError(
                "use '--api v2' option to decode with non-default language model"
            )
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(
                    word_rnnlm.predictor, rnnlm.predictor, word_dict, char_dict
                )
            )
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(
                    word_rnnlm.predictor, word_dict, char_dict
                )
            )

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    # load transducer beam search
    if hasattr(model, "is_rnnt"):
        if hasattr(model, "dec"):
            trans_decoder = model.dec
        else:
            trans_decoder = model.decoder
        joint_network = model.joint_network

        beam_search_transducer = BeamSearchTransducer(
            decoder=trans_decoder,
            joint_network=joint_network,
            beam_size=args.beam_size,
            nbest=args.nbest,
            lm=rnnlm,
            lm_weight=args.lm_weight,
            search_type=args.search_type,
            max_sym_exp=args.max_sym_exp,
            u_max=args.u_max,
            nstep=args.nstep,
            prefix_alpha=args.prefix_alpha,
            score_norm=args.score_norm,
        )

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = (
                    feat[0][0]
                    if args.num_encs == 1
                    else [feat[idx][0] for idx in range(model.num_encs)]
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with window size %d frames",
                        args.streaming_window,
                    )
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info(
                            "Feeding frames %d - %d", i, i + args.streaming_window
                        )
                        se2e.accept_input(feat[i: i + args.streaming_window])
                    logging.info("Running offline attention decoder")
                    se2e.decode_with_attention_offline()
                    logging.info("Offline attention decoder finished")
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with threshold value %d",
                        args.streaming_min_blank_dur,
                    )
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i: i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                elif hasattr(model, "is_rnnt"):
                    nbest_hyps = model.recognize(feat, beam_search_transducer)
                else:
                    if args.oracle_wts:
                        model.all_pms = all_oracle_wts[name]
                    temp, pm_scores, all_lpz, nbest_hyps = model.recognize(
                        feat, args, train_args.char_list, rnnlm
                    )

                    if args.pm_save_path:
                        all_pm_scores[name] = pm_scores
                    if args.posterior_save_path:
                        all_lpz_vals[name] = all_lpz
                args.temperature = temp
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )
        import pickle as pkl
        if args.pm_save_path:
            pkl.dump(all_pm_scores, open(args.pm_save_path, 'wb'))
        if args.posterior_save_path:
            pkl.dump(all_lpz_vals, open(args.posterior_save_path, 'wb'))
    else:

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = (
                    load_inputs_and_targets(batch)[0]
                    if args.num_encs == 1
                    else load_inputs_and_targets(batch)
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i: i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(
                        feats, args, train_args.char_list, rnnlm=rnnlm
                    )

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyp, train_args.char_list
                    )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )


def get_cmvn_2(cmvn_file):
    import subprocess
    import numpy as np
    shell_cmd = "copy-matrix --binary=false {:s} - ".format(cmvn_file)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8').split('\n')

    r_m = r[1].strip().split()
    r_v = r[2].strip().split()
    frame_num = float(r_m[-1])
    means = np.asarray([float(x) / frame_num for x in r_m[0:-1]], dtype=np.float32)
    var = np.asarray([float(x) / frame_num for x in r_v[0:-2]], dtype=np.float32)

    return means, var


def get_cmvn(cmvn_file):
    import subprocess
    import numpy as np
    shell_cmd = "copy-matrix --binary=false {:s} - ".format(cmvn_file)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8').split('\n')

    sum_x = r[1].strip().split()
    sum_x_sq = r[2].strip().split()
    frame_num = float(sum_x[-1])
    E_x = np.asarray([float(x) / frame_num for x in sum_x[0:-1]], dtype=np.float32)
    E_x_sq = np.asarray([float(x) / frame_num for x in sum_x_sq[0:-2]], dtype=np.float32)
    Var_x = E_x_sq - E_x ** 2

    return E_x, Var_x


def get_pca(pca_file):
    import subprocess
    import numpy as np
    shell_cmd = "copy-matrix --binary=false {:s} - ".format(pca_file)
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8').split('\n')

    mat = []
    for row in r[1:-2]:
        one_row = [[float(x) for x in row.split()[:-1]]]
        one_row = np.array(one_row, dtype=np.float32)
        mat.append(one_row)

    one_row = [[float(x) for x in r[-2].split()[:-2]]]
    one_row = np.array(one_row, dtype=np.float32)
    mat.append(one_row)

    mat = np.concatenate(mat, axis=0)
    return mat


def recog_joint_cl(args):
    """Decode with several parallel models with the given args using joint PM.

    Args:
        args (namespace): The program arguments.

    """
    from src.espnet_extensions.nets.e2e_asr_transformer_cl import E2E
    from src.nnet.models.VAE import nnetVAE, FF_VAE, FF_VAE_Rotation

    set_deterministic_pytorch(args)
    model_list = args.model_list.strip().split(',')
    all_models = []
    all_train_args = []
    for i in model_list:
        print(i)
        # asr_model, train_args = load_trained_model(i, training=False)
        asr_model, train_args = load_trained_model(i)
        asr_model.eval()
        assert isinstance(asr_model, ASRInterface)
        asr_model.recog_args = args
        all_models.append(asr_model)
        all_train_args.append(train_args)

    if args.pm_type == "joint_vae":
        logging.info("Using JOINT VAE performance monitor")
    elif args.pm_type == "enc_vae":
        logging.info("Using encoder VAE performance monitor")
    elif args.pm_type == "true_joint_vae":
        logging.info("Using TRUE JOINT VAE performance monitor")
    elif args.pm_type == "enc_vae_ff":
        logging.info("Using FF VAE (trained on ASR encoder output) performance monitor")
    else:
        raise ValueError("Invalid pm.type = {:s}".format(args.pm_type))

    if args.weight_combination == "standard":
        logging.info("Using STANDARD (first reduction in time, then softmax) way of computing score")
    elif args.weight_combination == "alternative":
        logging.info("Using ALTERNATIVE (first softmax per frame - over models - then mean in time) way of computing score")
    elif args.weight_combination == "per_frame":
        logging.info("Using PER FRAME combination way of computing score")
    elif args.weight_combination == "per_frame_logits":
        logging.info("Using PER FRAME weight derived from LOGITS to combine models")
    elif args.weight_combination == "per_frame_posteriors":
        logging.info("Using PER FRAME weight derived from POSTERIORS to compute score")
    else:
        logging.error("Wrong option for computing score (weight_combination).")
        raise ValueError("Wrong option for computing score (weight_combination), Use 'alternative', 'standard' 'per_frame', 'per_frame_logits' or 'per_frame_posteriors'.")

    if args.pm_save_path:
        all_pm_scores = {}
    if args.posterior_save_path:
        all_lpz_vals = {}

    if args.pm_list:
        pm_list = args.pm_list.strip().split(',')
    pm_list_enc = args.pm_list_enc.strip().split(',')
    enc_transform = args.enc_feat_transform.strip().split(',')
    all_pm_models = []
    all_pm_enc_models = []
    all_enc_transform = []

    if args.pm_type == "true_joint_vae":
        for i in pm_list:
            model_dict = torch.load(i, map_location=lambda storage, loc: storage)
            pm_model = nnetVAE(model_dict["feature_dim"], model_dict["encoder_num_layers"],
                               model_dict["decoder_num_layers"], model_dict["hidden_dim"], model_dict["bn_dim"],
                               dropout=0,
                               use_gpu=False)
            pm_model.load_state_dict(model_dict["model_state_dict"])
            pm_model.eval()
            all_pm_models.append(pm_model)


    elif args.pm_type == "joint_vae":
        for i in pm_list:
            model_dict = torch.load(i, map_location=lambda storage, loc: storage)
            pm_model = nnetVAE(model_dict["feature_dim"], model_dict["encoder_num_layers"],
                               model_dict["decoder_num_layers"], model_dict["hidden_dim"], model_dict["bn_dim"],
                               dropout=0,
                               use_gpu=False)
            pm_model.load_state_dict(model_dict["model_state_dict"])
            pm_model.eval()
            all_pm_models.append(pm_model)

        for i in pm_list_enc:
            model_dict = torch.load(i, map_location=lambda storage, loc: storage)
            pm_model = nnetVAE(model_dict["feature_dim"], model_dict["encoder_num_layers"],
                               model_dict["decoder_num_layers"], model_dict["hidden_dim"], model_dict["bn_dim"],
                               dropout=0,
                               use_gpu=False)
            pm_model.load_state_dict(model_dict["model_state_dict"])
            pm_model.eval()
            all_pm_enc_models.append(pm_model)


    elif args.pm_type == "enc_vae":
        logging.info("Using encoder VAE performance monitor")
        for i in pm_list_enc:
            model_dict = torch.load(i, map_location=lambda storage, loc: storage)
            model_dict['only_AE'] = False
            model_dict["unit_decoder_var"] = False
            if model_dict["only_AE"]:
                logging.info("Found the model to be only an auto-encoder")
            pm_model = nnetVAE(model_dict["feature_dim"], model_dict["encoder_num_layers"],
                               model_dict["decoder_num_layers"], model_dict["hidden_dim"], model_dict["bn_dim"],
                               dropout=0, use_gpu=False)
            pm_model.load_state_dict(model_dict["model_state_dict"])
            pm_model.eval()
            all_pm_enc_models.append(pm_model)

        all_pm_models = None

    elif args.pm_type == "enc_vae_ff":
        logging.info("Using FF VAE (trained on ASR encoder output) performance monitor")
        for i in pm_list_enc:
            model_dict = torch.load(i, map_location=lambda storage, loc: storage)
            device = "cpu"
            config = model_dict["config"]
            if config.rotation_vae:
                pm_model = FF_VAE_Rotation(config, device)
            else:
                pm_model = FF_VAE(config, device)

            pm_model.load_state_dict(model_dict["model_state_dict"])
            pm_model.eval()
            all_pm_enc_models.append(pm_model)

    for i in enc_transform:
        if args.enc_feat_transform_type == 'pca':
            temp = get_pca(i)
        elif args.enc_feat_transform_type == 'cmvn':
            temp = get_cmvn(i)
        all_enc_transform.append(temp)

    model = E2E(all_models, all_pm_models, all_pm_enc_models, all_enc_transform)

    if args.streaming_mode and "transformer" in all_train_args[0].model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")

    for idx, i in enumerate(all_models):
        logging.info(
            " Total parameter of the ASR model number " + str(idx + 1) + " = "
            + str(sum(p.numel() for p in i.parameters()))
        )

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError(
                "use '--api v2' option to decode with non-default language model"
            )
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(
                    word_rnnlm.predictor, rnnlm.predictor, word_dict, char_dict
                )
            )
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(
                    word_rnnlm.predictor, word_dict, char_dict
                )
            )

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    # load transducer beam search
    if hasattr(model, "is_rnnt"):
        if hasattr(model, "dec"):
            trans_decoder = model.dec
        else:
            trans_decoder = model.decoder
        joint_network = model.joint_network

        beam_search_transducer = BeamSearchTransducer(
            decoder=trans_decoder,
            joint_network=joint_network,
            beam_size=args.beam_size,
            nbest=args.nbest,
            lm=rnnlm,
            lm_weight=args.lm_weight,
            search_type=args.search_type,
            max_sym_exp=args.max_sym_exp,
            u_max=args.u_max,
            nstep=args.nstep,
            prefix_alpha=args.prefix_alpha,
            score_norm=args.score_norm,
        )

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = (
                    feat[0][0]
                    if args.num_encs == 1
                    else [feat[idx][0] for idx in range(model.num_encs)]
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with window size %d frames",
                        args.streaming_window,
                    )
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info(
                            "Feeding frames %d - %d", i, i + args.streaming_window
                        )
                        se2e.accept_input(feat[i: i + args.streaming_window])
                    logging.info("Running offline attention decoder")
                    se2e.decode_with_attention_offline()
                    logging.info("Offline attention decoder finished")
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with threshold value %d",
                        args.streaming_min_blank_dur,
                    )
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i: i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                elif hasattr(model, "is_rnnt"):
                    nbest_hyps = model.recognize(feat, beam_search_transducer)
                else:

                    temp, pm_scores, all_lpz, nbest_hyps = model.recognize(
                        feat, args, train_args.char_list, rnnlm
                    )

                    if args.pm_save_path:
                        all_pm_scores[name] = pm_scores
                    if args.posterior_save_path:
                        all_lpz_vals[name] = all_lpz
                args.temperature = temp
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )
        import pickle as pkl
        if args.pm_save_path:
            pkl.dump(all_pm_scores, open(args.pm_save_path, 'wb'))
        if args.posterior_save_path:
            pkl.dump(all_lpz_vals, open(args.posterior_save_path, 'wb'))
    else:

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = (
                    load_inputs_and_targets(batch)[0]
                    if args.num_encs == 1
                    else load_inputs_and_targets(batch)
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i: i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(
                        feats, args, train_args.char_list, rnnlm=rnnlm
                    )

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyp, train_args.char_list
                    )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
