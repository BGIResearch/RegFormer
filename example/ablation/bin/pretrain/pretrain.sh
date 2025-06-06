source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11
module load libs/nccl/2.18.3_cuda11
module load compilers/gcc/12.3.0


##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=NONE
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=600

##Config nnodes node_rank master_addr
NNODES=$1
HOSTFILE=$2
HOST=`hostname`
flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_IP=`head -n 1 ${HOSTFILE}`
echo $MASTER_IP

HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=${HOST_RANK}-1

DISTRIBUTED_ARGS="
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_IP \
    --master_port 30342 
 "
echo $DISTRIBUTED_ARGS
echo "
torchrun.sh ---------------
NNODES=${NNODES}, 
HOST=${HOST},
HOSTFILE=${HOSTFILE}, 
MASTER_IP=${MASTER_IP}, 
HOST_RANK=${HOST_RANK}, 
NODE_RANK=${NODE_RANK}
---------------------------"

# running
echo "Current Node:${HOST}"
echo "Number of Nodes: ${NNODES}"
echo "Current Node Rank: ${NODE_RANK}"

config_fille=$3
nproc_per_node=$4
torchrun --nproc_per_node=${nproc_per_node} --master_port=19932 --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} \
  /home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/downstream_task/pretrain_task_mamba.py $config_fille

