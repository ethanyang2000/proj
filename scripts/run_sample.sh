#!/bin/sh
task_type="collect"
scene_type="house"
scene='1b'
layout=0
num_agents=2
exp="debug"
seed=1

echo "task type is ${task_type}, scene type is ${scene_type}, scene is ${scene},\
layout is ${layout}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python run_env.py --num_agents ${num_agents} --task_type ${task_type} \
--scene_type ${scene_type} --scene ${scene} --layout ${layout} --seed ${seed} --episodes 10 --max_steps 100 \
--wandb_name 'ethanyang' --user_name 'yyx' --exp_name ${exp} --num_objects 4 --local_dir 'E:/tdw_lib/' --use_wandb

echo "training is done!"