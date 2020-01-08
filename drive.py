import multiprocessing
from random import randint
from time import sleep, time
from run import execute

loop = True
threads = multiprocessing.cpu_count()
cores = threads / 2

# (id, algorithm, env, episodes, timesteps, update_cadence=100, seed=100, lr=1e-2)
experiments = [
    ("1a", "DQN", "CartPole-v0", 500, 500, 10, 100, 1e-2),
    ("1b", "DQN", "CartPole-v0", 500, 500, 100, 100, 1e-2),
    ("2a", "DDQN", "CartPole-v0", 500, 500, 10, 100, 1e-2),
    ("2b", "DDQN", "CartPole-v0", 500, 500, 100, 100, 1e-2),
    ("3a", "VDQN", "CartPole-v0", 500, 500, 10, 100, 1e-2),
    ("3b", "VDQN", "CartPole-v0", 500, 500, 100, 100, 1e-2),
    ("4a", "DVDQN", "CartPole-v0", 500, 500, 10, 100, 1e-2),
    ("4b", "DVDQN", "CartPole-v0", 500, 500, 100, 100, 1e-2),

    ("5a", "DQN", "CartPole-v0", 500, 500, 100, 100, 1e-3),
    ("5b", "DQN", "CartPole-v0", 500, 500, 100, 100, 1e-3),
    ("6a", "DDQN", "CartPole-v0", 500, 500, 100, 100, 1e-3),
    ("6b", "DDQN", "CartPole-v0", 500, 500, 100, 100, 1e-3),
    ("7a", "VDQN", "CartPole-v0", 500, 500, 100, 100, 1e-3),
    ("7b", "VDQN", "CartPole-v0", 500, 500, 100, 100, 1e-3),
    ("8a", "DVDQN", "CartPole-v0", 500, 500, 10, 100, 1e-3),
    ("8b", "DVDQN", "CartPole-v0", 500, 500, 10, 100, 1e-3),
]

def run(id, algorithm, env, episodes, timesteps, update_cadence, seed, lr):
    start = time()
    name = "{}: {}-{}-{}".format(id, algorithm, env, episodes)
    print("Starting {}".format(name))
    execute(algorithm, env, episodes, timesteps, update_cadence, seed, lr, silent=True)
    print("Finished {} in {} seconds".format(name, time() - start))


def process_alive(proc):
    proc.join(timeout=0)
    if proc.is_alive():
        return True
    else:
        return False

num_processes = 2
processes = []
# available_affinities = list(range(cores))
next_experiment = 0

while(next_experiment < len(experiments)):
    # Remove completed processes.
    processes = list(filter(process_alive, processes))
    while(len(processes) < num_processes and next_experiment < len(experiments)):
        p = multiprocessing.Process(target=run, args=experiments[next_experiment])
        p.start()
        processes.append(p)
        next_experiment += 1
        if loop:
            next_experiment = next_experiment % len(experiments) # Loop forever
    sleep(0.5)

for p in processes:
    p.join()

print("Complete")