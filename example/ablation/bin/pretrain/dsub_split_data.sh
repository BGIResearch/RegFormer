#!/bin/bash
#DSUB -n regformer_split_blood_data
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=24;gpu=0;mem=100000"
#DSUB -oo ../logs/regformer_split_blood_data.out
#DSUB -eo ../logs/regformer_split_blood_data.err
##DSUB -pn "cyclone001-agent-156"


##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

##Config nnodes node_rank master_addr
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11
module load libs/nccl/2.18.3_cuda11
module load compilers/gcc/12.3.0

cd /home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/ablation;
python -u split_data.py