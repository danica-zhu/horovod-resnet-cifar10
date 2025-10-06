# Horovod ResNet18 CIFAR-10 Benchmark
This repository demonstrates distributed data-parallel training using Horovod with PyTorch, evaluated on ResNet18 + CIFAR-10.
We benchmark throughput, speedup, and parallel efficiency across 1/2/4× NVIDIA A16 GPUs, with different per-GPU batch sizes.

## Environment
+ Python 3.7
+ PyTorch 1.8.1 + CUDA 11.1
+ Horovod 0.22.1 (compiled with NCCL + Gloo)
+ torchvision, tqdm

## Training
**Single GPU**
```bash
horovodrun -np 1 -H localhost:1 \
    python src/train_hvd.py --epochs 5 --batch-size 512
```
**Multi GPU**
```bash
horovodrun -np 2 -H localhost:2 \
    python src/train_hvd.py --epochs 5 --batch-size 512

horovodrun -np 4 -H localhost:4 \
    python src/train_hvd.py --epochs 5 --batch-size 512
```

## Results

|**GPUs**|**Per-GPU Batch**|**Global Throughput (img/s)**|**Speedup**|**Efficiency**|**Val Acc (5 epochs)**|
|---|---|---|---|---|---|
|1|512|5643|1.00×|100%|61.1%|
|2|512|8139|1.44×|72%|58.6%|
|4|512|12342|2.19×|55%|55.4%|
|1|1024|6125|1.00×|100%|60.4%|
|2|1024|9845|1.61×|80%|54.3%|
|4|1024|15108|2.47×|62%|48.8%|

## Summary
+ Scaling to 4 GPUs achieved **2.47× speedup (62% efficiency)** at per-GPU batch 1024.
+ Larger batch size improved parallel efficiency but reduced validation accuracy.
+ Proposed remedies include **LR warmup**, **CosineAnnealing/OneCycleLR**, and **AMP** for faster training.
