# Exploring Variational Deep Q Networks

This study provides a research-ready implementation of [Tang and Kucukelbir's Variational Deep Q Network](https://arxiv.org/abs/1711.11225), a novel approach to maximising the efficiency of exploration in complex learning environments using Variational Bayesian Inference, using the [Edward PPL](http://edwardlib.org/). Alongside reference implementations of both Traditional and Double Deep Q Networks, a small novel contribution is presented --- the Double Variational Deep Q Network, which incorporates improvements to increase the stability and robustness of inference-based learning. Finally, an evaluation and discussion of the effectiveness of these approaches is discussed in the wider context of Bayesian Deep Learning.

The full report is available here: [Exploring VDQNs](https://github.com/HarriBellThomas/VDQN/blob/master/Exploring_VDQNs_Report.pdf).


### Using the Framework

These steps assume a fresh Ubuntu installation is being used. The process may deviate slightly if a different system is used. A helper script is included for installing the required dependencies on Linux; the process is near-identical for macOS.

```bash
git clone https://github.com/HarriBellThomas/VDQN.git
cd VDQN
./init.sh
source env/bin/activate
```

There are four main files to be aware of:

 - `run.py` --- this is the main entrypoint for running a single instance of one of the four algorithms. It accepts a number of CLI arguments for configuring the parameters it used.
 - `DQN.py` --- this is the source file containing the implementations for both DQN and DDQN.
 - `VDQN.py` --- this is the source file containing the implementations for both VDQN and DVDQN.
 - `drive.py` --- this script is the driver used for running experiments at scale. It constructs a collection of 80 experiments, and iteratively loops through them.

The `run.py` script can be used as follows:

```bash
python3 run.py
    --algorithm DQN|DDQN|VDQN|DVDQN \
    --environment CartPole-v0 \
    --episodes 200 \
    --timesteps 200 \
    --lossrate 1e-2
```