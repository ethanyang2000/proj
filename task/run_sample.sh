#!/bin/sh
task_type="Habitat"
scene_type="pointnav_gibson"
scene=
layout=
num_agents=2
exp="eval_new_frontier_hm3d_11"
seed=

echo "task type is ${task_type}, scene type is ${scene_type}, scene is ${scene},\
layout is ${layout}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python run_env.py --num_agents ${num_agents} --task_type ${task_type}\
--scene_type ${scene_type} --scene ${scene} --layout ${layout} --seed ${seed} --episodes 10 --max_steps 100
--log --wandb_name 'ethanyang' --user_name 'yyx' --exp_name ${exp}

echo "training is done!"