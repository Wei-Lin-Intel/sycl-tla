export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export PYTHONPATH=/home/linwei/sycl-tla/build_pyfmha/examples/06_bmg_flash_attention

seq=64
nhead=1

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

export sp=2
export ZE_AFFINITY_MASK=4,5,6,7

mpirun -n $sp -l \
  python ring_xpu_fa.py \
    --q-seq-len $seq \
    --q-nhead $nhead --kv-nhead $nhead \
    --qk-hdim 128 --v-hdim 128 \
    --warmup 2 --loops 10 --seed 2026

