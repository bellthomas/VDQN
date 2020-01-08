import multiprocessing
from random import randint
from time import sleep, time
from run import execute

loop = True
threads = multiprocessing.cpu_count()
cores = threads / 2

algorithms = ["DQN", "DDQN", "VDQN", "DVDQN"]
loss_rates = [1e-2, 1e-3, 1e-4]
environments = [
    ["CartPole-v0", 400, 200],
    ["CartPole-v1", 400, 500],
    ["MountainCar-v0", 400, 500],
    ["Acrobot-v1", 400, 500]
]

_i = 0
update_cadence = 10
seed = 100
experiments = []
for _e in environments:
    for _lr in loss_rates:
        for _a in algorithms:
            experiments.append((_i, _a, _e[0], _e[1], _e[2], update_cadence, seed, _lr))
            _i += 1


def run(id, algorithm, env, episodes, timesteps, update_cadence, seed, lr):
    start = time()
    name = "{}: {}-{}-{}".format(id, algorithm, env, episodes)
    print("Starting {}".format(name))
    execute(algorithm, env, episodes, timesteps, update_cadence, seed, lr, silent=True)
    print("Finished {} in {} seconds".format(name, time() - start))


def process_alive(proc):
    proc.join(timeout=0)
    return proc.is_alive()

num_processes = 2
processes = []
# available_affinities = list(range(cores))
next_experiment = 0 if not loop else random.ranint(0, len(experiments))

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