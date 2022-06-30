#!/usr/bin/env bash

. ./path.sh

stage=0
nj=10
hybrid_dir=exp/hybrid_generative_pytorch
data_dir=data
lang_dir=lang_test_bg
feat_type=mfcc
hmm_dir=exp/tri3
use_gpu=true
train_set=train
concat_train_set=train
dev_set=dev
concat_dev_set=dev
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
num_egs_jobs=2
egs_dir=
concat_egs_dir=
auto_resume=true


# Neural network config
encoder_num_layers=2
decoder_num_layers=2
classifier_num_layers=1
in_channels=1,32,64
out_channels=32,64,128
kernel=3,5
ae_num_layers=1
hidden_dim=256
bn_dim=100
batch_size=64
epochs=300
model_save_interval=10
weight_decay=0.001
vae_type=modulation
nopool=true
beta=1
nfilters=6
nrepeats=50
ar_steps=3,5
filt_type=ellip
reg_weight=0.1
bn_bits=16
use_transformer=false
optimizer='sgd'
lrr=0.9 # Learning rate reduction factor when val loss goes up
enc_var_init=
unit_decoder_var=false

# Feature config
feature_dim=
left_context=
right_context=
max_seq_len=512
ali_type="phone"
ali_append=
per_utt_cmvn=true
skip_cmvn=false
do_pca=false
out_dist='gauss'

. utils/parse_options.sh || exit 1;

if [ -z ${feature_dim} ] && [ -z ${egs_dir} ]; then
  feature_dim=`feat-to-dim scp:${data_dir}/${train_set}/feats.scp -`
fi

if [ -z ${egs_dir} ] ; then
    if [ -z ${feature_dim} ] ; then echo "Set feature_dim when providing egs_dir"; exit 1; fi
fi

mkdir -p $hybrid_dir
log_dir=$hybrid_dir/log


if [ $stage -le 0 ] && [ -z ${egs_dir} ]; then

  if $skip_cmvn; then
    echo "$0: No cmvn computed..."
  elif $per_utt_cmvn; then
    cmvn_type=cmvn_utt
    for x in $train_set $test_set $dev_set; do
      cmvn_path=`realpath $hybrid_dir/perutt_cmvn_${x}_${feat_type}`
      compute-cmvn-stats \
        scp:$data_dir/$x/feats.scp \
        ark,scp:$cmvn_path.ark,$cmvn_path.scp  || exit 1;
    done
  else
    cmvn_type=cmvn
    cmvn_path=`realpath $hybrid_dir/global_cmvn_${feat_type}`
    compute-cmvn-stats scp:$data_dir/$train_set/feats.scp $cmvn_path  || exit 1;
  fi

  if $skip_cmvn && $do_pca; then
    echo "$0: Computing a PCA transform for the training data"
    utils/shuffle_list.pl $data_dir/${train_set}/feats.scp | sort |\
      est-pca --normalize-mean=true --normalize-variance=true scp:- $hybrid_dir/pca_all.mat
  fi

  for x in $train_set $dev_set $test_set ; do
    egs_dir=$hybrid_dir/egs/$x
    mkdir -p $egs_dir

    if $skip_cmvn; then
      cmvn_opts=""
    elif $per_utt_cmvn; then
      cmvn_path=$hybrid_dir/perutt_cmvn_${x}_${feat_type}.scp
      cmvn_opts="--feat_type=$cmvn_type,$cmvn_path"
    else
      cmvn_path=$hybrid_dir/global_cmvn_${feat_type}
      cmvn_opts="--feat_type=$cmvn_type,$cmvn_path"
    fi

    if $skip_cmvn && $do_pca; then
      cmvn_opts="--feat_type=pca,$hybrid_dir/pca_all.mat"
    fi

    if [ ! -z ${left_context} ] && [ ! -z ${right_context} ] ; then
      cmvn_opts+=" --concat_feats=${left_context},${right_context}"
    fi

    if [ ! -z $ali_append ]; then
      ali_name=${ali_append}_${x}
    else
      ali_name=$x
    fi

    data_prep_for_seq.py $cmvn_opts\
      --num_jobs=$num_egs_jobs \
      --ali_type=$ali_type \
      --max_seq_len=$max_seq_len \
      $data_dir/$x/feats.scp \
      ${hmm_dir}_ali_${ali_name} \
      $egs_dir || exit 1;
  done
  egs_dir=$hybrid_dir/egs
fi

if [ -z ${egs_dir} ]; then
  egs_dir=$hybrid_dir/egs
fi

if [ ! -z ${concat_egs_dir} ]; then
  add_vae_opts="$add_vae_opts --concat_egs_dir=${concat_egs_dir} --concat_train_set=${concat_train_set} --concat_dev_set=${concat_dev_set}"
fi

if ${use_transformer}; then
   add_vae_opts="$add_vae_opts --use_transformer"
fi

if [ ! -z ${enc_var_init} ] ; then
   add_vae_opts="$add_vae_opts --enc_var_init=${enc_var_init}"
fi

if ${unit_decoder_var} ; then
   add_vae_opts="$add_vae_opts --unit_decoder_var"
fi

if [ $stage -le 1 ]; then
  exp_name=exp_1
  # Auto resume
  if ${auto_resume}; then
    if [ -d "${hybrid_dir}/${nn_name}/${exp_name}.dir/" ]; then
      last_model=`ls -t ${hybrid_dir}/${nn_name}/${exp_name}.dir/*.model | head -n1`
      add_vae_opts="${add_vae_opts} --resume_checkpoint=${last_model}"
      echo "Resuming training from checkpoint ${last_model}"
    else
      echo "No checkpoints found to resume training, starting training from epoch 0"
    fi
  fi

  if $use_gpu; then
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_VAE_${nn_name}.log \
      $vae_script $add_vae_opts \
      --use_gpu \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --optimizer=$optimizer \
      --feature_dim=$feature_dim \
      --out_dist=$out_dist \
      --lrr=$lrr \
      --model_save_interval=$model_save_interval \
      --experiment_name=${exp_name} \
      $egs_dir \
      $hybrid_dir/$nn_name || exit 1;
  else

    queue.pl --mem 5G \
      $hybrid_dir/log/train_VAE_${nn_name}.log \
      $vae_script $add_vae_opts\
      --train_set=$train_set \
      --dev_set=$dev_set \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --optimizer=${optimizer} \
      --feature_dim=$feature_dim \
      --out_dist=$out_dist \
      --lrr=$lrr \
      --model_save_interval=$model_save_interval \
      --experiment_name=${exp_name} \
      $egs_dir \
      $hybrid_dir/$nn_name || exit 1;
  fi
fi

