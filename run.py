import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf

from AlgorithmConfig import AlgorithmConfig
from DQN import DQN
from VDQN import VDQN

# Envs: ["MountainCar-v0", "CartPole-v0", "CartPole-v1", "Acrobot-v1", "Tennis-v0", "AsterixNoFrameskip-v4", "Asteroids-v0"]

def execute(algorithm, env, episodes, timesteps, update_cadence=100, seed=100, lr=1e-2, epsilon=5000, gamma=0.99, silent=False):
    # Initialise
    algorithm = algorithm.upper()
    if not algorithm in ["DQN", "DDQN", "VDQN", "DVDQN"]:
        sys.exit("Invalid algorithm")
    
    os.environ['CHAINER_SEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Logs
    output_dir = "logs/{}/{}/l{}_g{}".format(
        algorithm, env, lr, gamma
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir_abs = output_path.absolute().as_posix()

    output_file_index = 0
    output_file = Path("{}/{}-{}".format(output_dir_abs, output_file_index, episodes))
    while output_file.exists():
        output_file_index += 1
        output_file = Path("{}/{}-{}".format(output_dir_abs, output_file_index, episodes))
    output_file_uri = output_file.absolute().as_posix()

    # Build concifg object
    config = AlgorithmConfig({
        "output_path": output_dir_abs,
        "episodes": episodes,
        "environment": env,
        "post_episode": lambda x: handlePostEpisode(x, output_file_uri, silent, variational=(algorithm in ["VDQN","DVDQN"])),
        "maximum_timesteps": timesteps,
        "network_update_frequency": update_cadence,
        "epsilon_decay_period": epsilon,
        "loss_rate": lr,
        "gamma": gamma,
    })

    switcher = {
        "DQN": lambda: DQN(config).run(),
        "DDQN": lambda: DQN(config, double=True).run(),
        "VDQN": lambda: VDQN(config).run(),
        "DVDQN": lambda: VDQN(config, double=True).run(),
    }
    func = switcher.get(algorithm, lambda: sys.exit("No algorithm: {}".format(algorithm)))
    func()

def handlePostEpisode(data, output, silent, variational=False):
    dataline = ("Episode {0} (i: {1}, {2} seconds) --- r: {3} (avg: {4}){5}".format(
        data.get("episode", "-1"),
        data.get("iteration", "-1"),
        "{:.2f}".format(data.get("duration", -1)),
        data.get("reward", "-1"),
        data.get("meanPreviousRewards", "-1"),
        "" if not variational else " (vi: {}, bellman: {})".format(
            data.get("variationalLosses", "-1"), data.get("bellmanLosses", "-1"),
        ),
    ))

    if not silent:
        print(dataline)
    with open(output, "a+") as log:
        log.write("{}\n".format(dataline))

def main():
    argparser = ArgumentParser(description='VDQN/DQN Demonstrator')
    argparser.add_argument('--algorithm', '-a', type=str, default='DQN', help='Algorithm to run')
    argparser.add_argument('--environment', type=str, default='CartPole-v0', help='OpenAI Gym Environment')
    argparser.add_argument('--episodes', '-e', type=int, default=100, help='Duration (episodes)')
    argparser.add_argument('--timesteps', '-t', type=int, default=500, help='Max episode length (iterations)')
    argparser.add_argument('--updates', '-u', type=int, default=100, help='Active network update cadence (iterations)')
    argparser.add_argument('--lossrate', '-l', type=float, default=1e-2, help='Loss rate')
    argparser.add_argument('--decay', '-d', type=float, default=5000, help='Epsilon decay period (iterations)')
    args = argparser.parse_args()
    execute(args.algorithm, args.environment, args.episodes, args.timesteps, update_cadence=args.updates, lr=args.lossrate, epsilon=args.decay)

if __name__ == '__main__':
    main()
