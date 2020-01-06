#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Harri Bell-Thomas
# Date: January 2020
# License: https://github.com/HarriBellThomas/VDQN/blob/master/LICENSE

import copy
from collections import deque

import chainer
import gym
import numpy as np
from chainer import Chain, functions, links, optimizers
from time import time

from helpers.ReplayBuffer import ReplayBuffer


class DQN:

    ###
    class QFunction(Chain):
        def __init__(self, obsSpace, actSpace, hiddenLayers=100):
            super(DQN.QFunction, self).__init__()
            with self.init_scope():
                self.l0 = links.Linear(obsSpace, hiddenLayers)
                self.l1 = links.Linear(hiddenLayers, hiddenLayers)
                self.l2 = links.Linear(hiddenLayers, actSpace)

        def __call__(self, x):
            """Compute Q-values of actions for given observations."""
            h = functions.relu(self.l0(x))
            h = functions.relu(self.l1(h))
            return self.l2(h)
        

    def __init__(self, config, double=False, debug=False):
        self.__enableDebug = debug
        self.__config = config
        self.__doubleDQN = double

    def __debug(self, msg : str, newlines: int = 0):
        if(self.__enableDebug):
            print("{0}DQN: {1}".format(
                "".join(["\n" for i in range(newlines)]),
                msg
            ))

    def __update(self, _q, _qTarget, optimiser, samples, gamma=0.99):
        """Update a Q-function with given samples and a target Q-function."""
        # self.__debug("Running update...")

        currentStates = _q.xp.asarray(samples["states"], dtype=np.float32)
        actions = _q.xp.asarray(samples["actions"], dtype=np.int32)
        rewards = _q.xp.asarray(samples["rewards"], dtype=np.float32)
        completes = _q.xp.asarray(samples["completes"], dtype=np.float32)
        nextState = _q.xp.asarray(samples["nextStates"], dtype=np.float32)

        # Predicted values: Q(s,a)
        predictions = functions.select_item(_q(currentStates), actions)

        # Target values: r + gamma * max_b Q(s',b)
        with chainer.no_backprop_mode():
            if self.__doubleDQN:
                _qNext = functions.select_item(
                    _qTarget(nextState),
                    functions.argmax(_q(nextState), axis=1)
                )
            else:
                _qNext = functions.max(_qTarget(nextState), axis=1)

            target = rewards + gamma * (1 - completes) * _qNext

        loss = functions.mean(
            functions.huber_loss(predictions, target, delta=1.0, reduce='no')
        )
        _q.cleargrads()
        loss.backward()
        optimiser.update()

    def __greedyAction(self, _q, state):
        """Get a greedy action wrt a given Q-function."""
        state = _q.xp.asarray(state[None], dtype=np.float32)
        with chainer.no_backprop_mode():
            _qValue = _q(state).data[0]
        return int(_qValue.argmax())


    def run(self):
        self.__debug(self.__config.get("episodes"))

        # Build OpenAI Gym environment
        environment = gym.make(self.__config.get("environment"))
        if hasattr(environment, 'env'):
            environment = environment.env
        obvSpace = environment.observation_space.low.size
        actSpace = environment.action_space.n

        # Get parameter configuration.
        replayStartThreshold = self.__config.get("replay_start_threshold")
        minimumEpsilon = self.__config.get("minimum_epsilon")
        epsilonDecayPeriod = self.__config.get("epsilon_decay_period") # Iterations
        rewardScaling = self.__config.get("reward_scaling")
        minibatchSize = self.__config.get("minibatch_size")
        hiddenLayers = self.__config.get("hidden_layers")
        gamma = self.__config.get("gamma")
        networkUpdateFrequency = self.__config.get("network_update_frequency")
        maximumNumberOfSteps = self.__config.get("maximum_timesteps")
        
        # Initialise storage queues.
        replayBuffer = ReplayBuffer(capacity = 10**6)
        episodeTotals = deque(maxlen = self.__config.get("episode_history_averaging"))  


        # Build the Q function modeller and optimiser.
        _q = self.QFunction(obvSpace, actSpace, hiddenLayers=hiddenLayers)
        _qTarget = copy.deepcopy(_q)
        optimiser = optimizers.Adam(eps=self.__config.get("loss_rate"))
        optimiser.setup(_q)

        # Episode loop.
        iteration = 0
        self.__debug("Starting episode loop...")
        for episode in range(self.__config.get("episodes")):
            self.__debug("\n\n--- EPISODE {} ---".format(episode))
            currentState = environment.reset()
            episodeRewards = 0
            running = True
            timestep = 0
            start = time()

            # Run an iteration of the current episode.
            while running and timestep < maximumNumberOfSteps:
                self.__debug("Episode {}: Timestep {}".format(episode, timestep))
                # Decay the epsilon value as the episode progresses.
                epsilon = 1.0
                if(len(replayBuffer) >= replayStartThreshold):
                    epsilon = max(
                        minimumEpsilon,
                        np.interp(
                            iteration,
                            [0, epsilonDecayPeriod],
                            [1.0, minimumEpsilon]
                        )
                    )

                # Select action to perform.
                # Either random or greedy depending on the current epsilon value.
                action = environment.action_space.sample() \
                    if np.random.rand() < epsilon \
                    else self.__greedyAction(_q, currentState)

                # Execute the chosen action.
                nextState, reward, completed, _ = environment.step(action)
                episodeRewards += reward

                # Save the experience.
                experience = ReplayBuffer.Experience(
                    currentState, action, reward * rewardScaling, completed, nextState
                )
                replayBuffer.add(experience)
                currentState = nextState

                # Sample and replay minibatch if threshold reached.
                if(len(replayBuffer) >= replayStartThreshold):
                    minibatchSamples = replayBuffer.randomSample(minibatchSize)
                    self.__update(_q, _qTarget, optimiser, minibatchSamples, gamma=gamma)


                # Update the target Q network.
                if iteration % networkUpdateFrequency == 0:
                    _qTarget = copy.deepcopy(_q)

                iteration += 1
                timestep += 1
                running = not completed

            # Run post episode event handler.
            episodeTotals.append(episodeRewards)
            self.__config.get("post_episode")({
                "episode": episode,
                "iteration": iteration,
                "reward": episodeRewards,
                "meanPreviousRewards": np.mean(episodeTotals),
                "duration": time() - start,
            })
