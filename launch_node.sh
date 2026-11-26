#!/usr/bin/env bash
# Author: 数萌AI服务器 (winintel)
set -euo pipefail
root="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
node_rank="${1:?Usage: launch_node.sh NODE_RANK [generation arguments]}"; shift
: "${WININTEL_REFERENCE_ENV:?Set an isolated Python environment}"
: "${WININTEL_IB_HCA:?Set the active InfiniBand HCA and port}"
: "${WININTEL_SOCKET_IFACE:?Set the bootstrap network interface}"
: "${WININTEL_MASTER:?Set the rendezvous host address}"
export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_NET_GDR_LEVEL=0
export NCCL_IB_HCA="$WININTEL_IB_HCA"
export NCCL_SOCKET_IFNAME="$WININTEL_SOCKET_IFACE"
export GLOO_SOCKET_IFNAME="$WININTEL_SOCKET_IFACE"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="$root/nccl.%h.%p.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
chmod +x "$root/run_reference.sh"
exec bash "$root/run_reference.sh" -m torch.distributed.run \
    --nnodes "${WININTEL_NNODES:-2}" --nproc-per-node "${WININTEL_GPUS:-8}" \
    --node-rank "$node_rank" --master-addr "$WININTEL_MASTER" \
    --master-port "${WININTEL_MASTER_PORT:-29561}" \
    --no-python "$root/run_reference.sh" "$root/generate.py" "$@"
