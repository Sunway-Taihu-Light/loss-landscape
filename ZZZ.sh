# 创建 activate.d 目录

mkdir -p $HOME/miniconda3/envs/LossEnv/etc/conda/activate.d

# 写入环境变量
cat > $HOME/miniconda3/envs/LossEnv/etc/conda/activate.d/env_vars.sh <<EOF
#!/bin/bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EOF
echo 'finished'