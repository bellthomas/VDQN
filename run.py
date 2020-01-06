from argparse import ArgumentParser
import sys, os
import numpy as np
from pathlib import Path
from VDQN import VDQN
from DQN import DQN
from AlgorithmConfig import AlgorithmConfig

def main():
    argparser = ArgumentParser(description='VDQN/DQN Demonstrator')
    argparser.add_argument('--algorithm', '-a', type=str, default='DQN', help='Algorithm to run')
    argparser.add_argument('--environment', type=str, default='CartPole-v0', help='OpenAI Gym Environment')
    argparser.add_argument('--episodes', '-e', type=int, default=100, help='Duration (episodes)')
    args = argparser.parse_args()

    # Initialise
    algorithm = args.algorithm.upper()
    if not algorithm in ["DQN", "DDQN", "VDQN"]:
        sys.exit("Invalid algorithm")
    
    loss_rate = 1e-2
    seed = 100
    os.environ['CHAINER_SEED'] = str(seed)
    np.random.seed(seed)

    # Logs
    output_dir = "{}/{}/loss_{}_episodes_{}".format(
        algorithm, args.environment, loss_rate, args.episodes
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir_abs = output_path.absolute().as_posix()

    # Build concifg object
    config = AlgorithmConfig({
        "output_path": output_dir_abs,
        "episodes": args.episodes,
        "environment": args.environment,
        "post_episode": lambda x: handlePostEpisode(x)
    })

    switcher = {
        "DQN": lambda: DQN(config).run(),
        "DDQN": lambda: DQN(config, double=True).run(),
        "VDQN": lambda: print("VDQN"),
    }
    func = switcher.get(algorithm, lambda: sys.exit("No algorithm: {}".format(algorithm)))
    func()

def handlePostEpisode(data):
    print("Episode {0} (i: {1}) --- r: {2} (avg: {3})".format(
        data.get("episode", "-1"),
        data.get("iteration", "-1"),
        data.get("reward", "-1"),
        data.get("meanPreviousRewards", "-1")
    ))

if __name__ == '__main__':
    main()