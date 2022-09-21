from env.env import TaskEnv
import sys
import os
import wandb
import argparse
from pathlib import Path
import socket
import setproctitle

def make_env(all_args):
    return TaskEnv(args=all_args)

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    # for simulator
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--launch_build', default=True, action='store_false')
    parser.add_argument('--num_agents', default=2, type=int)
    parser.add_argument('--exp_type', default='sample', type=str)

    parser.add_argument('--task_type', default='collect', type=str)
    parser.add_argument('--scene_type', default='house', choices=['kitchen', 'house'])
    parser.add_argument('--scene', default='1b', choices=['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c'])
    parser.add_argument('--layout', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--num_objects', default=4, type=int)
    parser.add_argument('--local_dir', default=None, type=str)

    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--use_wandb', default=True, action='store_false')
    parser.add_argument('--wandb_name', default='ethanyang', type=str)
    parser.add_argument('--user_name', default='ethanyang', type=str)
    parser.add_argument('--exp_name', default='debug', type=str)

    
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    all_args = parse_args(args)

    print("scene type is {} with the scene {} and layout {}. \n".format(all_args.scene_type, all_args.scene, all_args.layout))

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.exp_type / all_args.task_type / all_args.scene_type / all_args.scene
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project='tdw',
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.task_type) + "_" + str(all_args.scene_type) + "_" +
                         str(all_args.scene) +
                         "_seed" + str(all_args.seed),
                         group=all_args.exp_type,
                         dir=str(run_dir),
                         job_type="debug",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
                              str(all_args.env_name) + "-" + str(all_args.exp_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_env(all_args)

    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.exp_type == 'sample':
        from runner.runners import SampleRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])
