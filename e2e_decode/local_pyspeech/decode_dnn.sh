#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh

stage=0
nj=10
nj_llhood=1
# Model
model=model
graph_name=graph
hmm_dir=exp/tri3

# Data and lang
data_dir=data    # Kaldi data dir
test_set='test'     # Test set name under kaldi dir
lang_dir=data/lang_test_bg  # Kaldi lang dir
pt_files_dir=pt_files


# Decoding
decode_dir=decode
prior_weight=0.2
num_threads=8
min_active=200
max_active=7000
beam=13
lattice_beam=8
prior_file=prior
score_script=score_wsj.sh
# Others
remove_ll=false # Remove loglikelihood directory after decoding
append='martinku'
add_opts=
decode_config=conf/hybrid_decode_basic.conf
. utils/parse_options.sh

if [ ! -z ${decode_config} ]; then source ${decode_config}  ; fi #NOTE: Overwrites stuff from command line!

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"


decode_dir="${decode_dir}"
#echo $cuda_cmd
#which queue-freegpu.pl

ll_dir=${decode_dir}/loglikelihoods_${append}

log_dir=${decode_dir}/log
mkdir -p ${ll_dir}
mkdir -p ${log_dir}
mkdir -p ${decode_dir}


if [ $stage -le 0 ]; then
  echo "$0: Compute Log-likelihood"

  ## Split .pt files into sub-folders
  python3 ../../src/decode_utils/split_pt_files.py --nj=${nj_llhood} ${pt_files_dir} ${store_files_dir} || exit 1;
  if [ ${nj_llhood} -le 5 ] ; then
    # Assuming that forward pass is done on GPU
    ${cuda_cmd} JOB=1:${nj_llhood} \
    ${log_dir}/compute_llikelihood_${append}.JOB.log \
    sleep JOB JOB JOB JOB JOB JOB JOB\; python3 ../../src/nnet/bin/get_dnn_posteriors.py ${add_opts}\
    --prior=${prior_file} \
    --prior_weight=${prior_weight} \
    $model \
    ${store_files_dir}/JOB \
    ${ll_dir}/llhoods.JOB.ll || exit 1;
  else
    queue.pl --mem 50G JOB=1:${nj_llhood} \
      ${log_dir}/compute_llikelihood_${append}.JOB.log \
      python3 ../../src/nnet/bin/get_dnn_posteriors.py --cpu ${add_opts}\
      --prior=${prior_file} \
      --prior_weight=${prior_weight} \
      $model \
      ${store_files_dir}/JOB \
      ${ll_dir}/llhoods.JOB.ll || exit 1;
  fi


  for n in `seq ${nj_llhood}`; do
   cat ${ll_dir}/llhoods.$n.ll.scp
  done > ${ll_dir}/all_llhoods
fi


for acwt in 0.05 0.1 0.2 0.4 ; do
  echo "Acoustic Model weight: ${acwt}"
  append=${acwt}
  decode_subdir=${decode_dir}/decode_${append}
  mkdir -p ${decode_subdir}
  if [ $stage -le 1 ]; then
    echo "$0: Make graph and Decode "
    utils/mkgraph.sh \
      ${lang_dir} ${hmm_dir} ${hmm_dir}/${graph_name} || exit 1;

    split_scp=""
    for n in `seq $nj`; do
      split_scp="${split_scp} ${log_dir}/llhood_split.$n.scp"
    done
    utils/split_scp.pl ${ll_dir}/all_llhoods ${split_scp} || exit 1;

    queue.pl --mem 2G --num-threads $num_threads JOB=1:$nj \
      $log_dir/decode_${append}.JOB.log \
      latgen-faster-mapped${thread_string} --min-active=$min_active \
      --max-active=${max_active} \
      --beam=$beam \
      --lattice-beam=${lattice_beam} \
      --acoustic-scale=$acwt \
      --allow-partial=true \
      --word-symbol-table=${hmm_dir}/${graph_name}/words.txt \
      ${hmm_dir}/final.mdl \
      ${hmm_dir}/${graph_name}/HCLG.fst \
      scp:${log_dir}/llhood_split.JOB.scp \
      "ark:|gzip -c > ${decode_subdir}/lat.JOB.gz" || exit 1;

    echo $nj > ${decode_subdir}/num_jobs
  fi

  if [ $stage -le 2 ]; then
    echo "$0: get WER "

    bash local_pyspeech/$score_script \
      --cmd "queue.pl" \
      --min-lmwt 1 \
      --max-lmwt 10 \
      ${data_dir}/${test_set} \
      ${hmm_dir}/${graph_name} \
      ${decode_subdir} || exit 1;
  fi
done

if $remove_ll; then
  echo "$0: Removing all files from log-likelihood directory"
  rm -r ${ll_dir}
fi
