# mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 \
# --model_file cifar10/trained_nets/checkpoint.pth \
# --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot

mpirun -n 2 python plot_surface.py --mpi --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 \
--model_file cifar10/trained_nets/checkpoint.pth \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot




# # 检查 mpirun 路径（现在应指向 Conda 环境）
# which mpirun
# # 输出应为: /root/miniconda3/envs/loss/bin/mpirun

# # 测试 Python 导入
# python -c "from mpi4py import MPI; print('Success!')"

# # 测试多进程
# mpirun -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"





# 创建 activate.d 目录

# mkdir -p $HOME/miniconda3/envs/LossEnv/etc/conda/activate.d

# # 写入环境变量
# cat > $HOME/miniconda3/envs/LossEnv/etc/conda/activate.d/env_vars.sh <<EOF
# #!/bin/bash
# export OMPI_ALLOW_RUN_AS_ROOT=1
# export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# EOF
# echo 'finished'

# # 重新激活环境生效
# conda deactivate
# conda activate LossEnv