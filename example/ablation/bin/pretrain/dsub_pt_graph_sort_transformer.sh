#!/bin/bash
#DSUB -n regformer_pt_graph_sort_transformer
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=24;gpu=4;mem=80000"
#DSUB -oo ../logs/pt/regformer_pt_graph_sort_transformer.out
#DSUB -eo ../logs/pt/regformer_pt_graph_sort_transformer.err

## Set scripts
RANK_SCRIPT="./pretrain.sh"

nproc_per_node=4

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/ablation/bin/"

## Set NNODES
NNODES=1

## Create nodefile

JOB_ID=${BATCH_JOB_ID}
NODEFILE=/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/logs/pt/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
config_file=/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/ablation/conf/pretrain/pt_graph_sort_transformer.toml

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE} $config_file $nproc_per_node
