# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# print(f"Hello from rank {comm.Get_rank()} of {comm.Get_size()}")
import torch

ckpt = torch.load("./cifar10/trained_nets/checkpoint.pth", weights_only=False)
print(ckpt.keys())
# 如果输出 dict_keys(['model', 'optimizer', 'epoch', 'args']) → 说明是 checkpoint