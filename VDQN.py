from collections import deque
from time import time

import gym
import numpy as np
from chainer import functions

import edward
import edward.models as edm
import tensorflow as tf
from helpers.ReplayBuffer import ReplayBuffer


class VDQN:

    class VariationalQFunction:
        def __init__(self, stateSpace, actionSpace, hiddenLayers, session,
                     optimiser=tf.train.AdamOptimizer(1e-3), scope='alpha',
                     parameterConfig={}):
            
            self.__stateSpace = stateSpace
            self.__actionSpace = actionSpace
            self.__hiddenLayers = hiddenLayers
            self.__nnLayers = [stateSpace] + hiddenLayers + [actionSpace]
            self.__session = session
            self.__optimiser = optimiser

            self.__tau = parameterConfig.get("tau", 1.0)
            self.__sigma = parameterConfig.get("sigma", 0.1)
            self.__sigmaRho = parameterConfig.get("sigmaRho", None)
            
            _prior_W_sigma = parameterConfig.get("prior_W_sigma", None)
            _prior_b_sigma = parameterConfig.get("prior_b_sigma", None)

            with tf.variable_scope(scope):
                self.__prior(_prior_W_sigma, _prior_b_sigma)
                self.__model()
                self.__posterior()
                self.__forwardComputation()
                self.__inference()
                self.__assignments_init()


        def __model(self):
            with tf.variable_scope("prior", reuse=True):
                stateX = tf.placeholder(tf.float32, [None, self.__stateSpace], name="stateX")
                actionX = tf.placeholder(tf.int32, [None], name="actionX")
                selected_actionX = tf.one_hot(actionX, self.__actionSpace, dtype=tf.float32)
                
                activation = tf.nn.relu(tf.matmul(stateX, self.__W[0]) + self.__b[0])
                layers = len(self.__W.keys())
                for i in range(1, layers-1):
                    activation = tf.nn.relu(tf.matmul(activation, self.__W[i]) + self.__b[i])
                activation = tf.matmul(activation, self.__W[layers-1]) + self.__b[layers-1]
                
                chosenAction = tf.reduce_sum(
                    tf.multiply(activation, selected_actionX),
                    axis=1
                )
                chosenActionDistribution = edm.Normal(loc=chosenAction, scale=self.__sigma, name="Y")

                self.__stateX = stateX
                self.__actionX = actionX
                self.__nextAction = chosenActionDistribution

        def __forwardComputation(self):
            self.__observation = tf.placeholder(tf.float32, [None, self.__stateSpace])
            noise_W, noise_b = {}, {}
            theta_W, theta_b = {}, {}

            for _i, W in self.__posterior_W.items():
                noise_W[_i] = tf.placeholder(tf.float32, [None] + list(W.shape))
                theta_W[_i] = self.__posterior_W_mu[_i] + tf.nn.softplus(self.__posterior_W_rho[_i]) * noise_W[_i]
            
            for _i, b in self.__posterior_b.items():
                noise_b[_i] = tf.placeholder(tf.float32, [None] + list(b.shape))
                theta_b[_i] = self.__posterior_b_mu[_i] + tf.nn.softplus(self.__posterior_b_rho[_i]) * noise_b[_i]

            layers = len(theta_W.keys())
            activation = tf.nn.relu(tf.einsum('ij,ijk->ik', self.__observation, theta_W[0]) + theta_b[0])
            for _i in range(1, layers-1):
                activation = tf.nn.relu(tf.einsum('ij,ijk->ik', activation, theta_W[_i]) + theta_b[_i])
            activation = tf.nn.relu(tf.einsum('ij,ijk->ik', activation, theta_W[layers-1]) + theta_b[layers-1])

            self.__Q_mu = activation
            self.__noise_W = noise_W
            self.__noise_b = noise_b
            self.__theta_W = theta_W
            self.__theta_b = theta_b


        def __prior(self, W_sigma, b_sigma):
            with tf.variable_scope("prior"):
                nnLayers = self.__nnLayers
                _numTransitions = len(nnLayers) - 1
                _defaultTransitionWeights = _numTransitions * [0.1]
                W_sigma = W_sigma or _defaultTransitionWeights[:]
                b_sigma = b_sigma or _defaultTransitionWeights[:]

                _index = 0
                _W, _b = {}, {}
                _shape_W, _shape_b = [], []
                layerTransitionPairings = zip(nnLayers[:-1], nnLayers[1:])
                for _left, _right in layerTransitionPairings:
                    _W[_index] = edm.Uniform(
                        low = tf.ones([_left, _right]) * (-10_000),
                        high = tf.ones([_left, _right]) * (10_000)
                    )
                    _b[_index] = edm.Uniform(
                        low = tf.ones(_right) * (-10_000),
                        high = tf.ones(_right) * (10_000)
                    )

                    _shape_W.append([_left, _right])
                    _shape_b.append([_right])
                    _index += 1

                self.__W = _W
                self.__b = _b
                self.__shape_W = _shape_W
                self.__shape_b = _shape_b

        def __posterior(self):
            with tf.variable_scope("posterior"):
                nnLayers = self.__nnLayers
                layerTransitionPairings = zip(nnLayers[:-1], nnLayers[1:])
                _index = 0
                sigmaRho = self.__sigmaRho or np.log(np.exp(0.017)-1.0)

                # Posterior collections.
                self.__posterior_W, self.__posterior_b = {}, {}
                self.__posterior_W_mu, self.__posterior_b_mu = {}, {}
                self.__posterior_W_rho, self.__posterior_b_rho = {}, {}

                for _left, _right in layerTransitionPairings:
                    with tf.variable_scope("posterior_W{}".format(_index)):
                        _width = np.sqrt(3 / _left)
                        self.__posterior_W_mu[_index] = tf.Variable(
                            tf.random_uniform([_left, _right], -1*_width, _width),
                            name="mean"
                        )
                        self.__posterior_W_rho[_index] = tf.Variable(
                            tf.random_uniform([_left, _right], sigmaRho, sigmaRho),
                            name="std", trainable=True
                        )
                        self.__posterior_W[_index] = edm.Normal(
                            loc = self.__posterior_W_mu[_index],
                            scale = tf.nn.softplus(self.__posterior_W_rho[_index])
                        )

                    with tf.variable_scope("posterior_b{}".format(_index)):
                        self.__posterior_b_mu[_index] = tf.Variable(
                            tf.random_uniform([_right], 0, 0),
                            name="mean"
                        )
                        self.__posterior_b_rho[_index] = tf.Variable(
                            tf.random_uniform([_right], sigmaRho, sigmaRho),
                            name="std", trainable=True
                        )
                        self.__posterior_b[_index] = edm.Normal(
                            loc = self.__posterior_b_mu[_index],
                            scale = tf.nn.softplus(self.__posterior_b_rho[_index])
                        )

                    _index += 1


        def __inference(self, iterations=2000):
            latentVariables = {}
            for x in self.__W.keys():
                latentVariables[self.__W[x]] = self.__posterior_W[x]
            for y in self.__b.keys():
                latentVariables[self.__b[y]] = self.__posterior_b[y]

            self.__actionTargets = tf.placeholder(tf.float32, [None])
            self.__inference = edward.KLqp(latentVariables, data={ self.__nextAction: self.__actionTargets })
            self.__inference.initialize(
                optimizer = self.__optimiser,
                scale = { self.__nextAction: 1 },
                n_iter = iterations
            )

        def __assignments_init(self):
            self.__values_W_mu = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_W_mu.items() }
            self.__values_W_rho = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_W_rho.items() }
            self.__values_b_mu = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_b_mu.items() }
            self.__values_b_rho = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_b_rho.items() }
            self.__assignments = []

            layers = len(self.__values_W_mu.keys())
            _t = self.__tau
            for _i in range(layers):
                self.__assignments.append(self.__posterior_W_mu[_i].assign(
                    _t * self.__values_W_mu[_i] + (1-_t) * self.__posterior_W_mu[_i]
                ))
                self.__assignments.append(self.__posterior_W_rho[_i].assign(
                    _t * self.__values_W_rho[_i] + (1-_t) * self.__posterior_W_rho[_i]
                ))
                self.__assignments.append(self.__posterior_b_mu[_i].assign(
                    _t * self.__values_b_mu[_i] + (1-_t) * self.__posterior_b_mu[_i]
                ))
                self.__assignments.append(self.__posterior_b_rho[_i].assign(
                    _t * self.__values_b_rho[_i] + (1-_t) * self.__posterior_b_rho[_i]
                ))

        def assign(self, W_mu, W_rho, b_mu, b_rho):
            variables = {}
            for i in range(len(self.__posterior_W.keys())):
                variables[self.__values_W_mu[i]] = W_mu[i]
                variables[self.__values_W_rho[i]] = W_rho[i]
                variables[self.__values_b_mu[i]] = b_mu[i]
                variables[self.__values_b_rho[i]] = b_rho[i]
            self.__session.run(self.__assignments, feed_dict=variables)

        def get_assignments(self):
            W_mu, W_rho, b_mu, b_rho = {}, {}, {}, {}
            for i in range(len(self.__posterior_W.keys())):
                W_mu[i] = self.__session.run(self.__posterior_W_mu[i])
                W_rho[i] = self.__session.run(self.__posterior_W_rho[i])
                b_mu[i] = self.__session.run(self.__posterior_b_mu[i])
                b_rho[i] = self.__session.run(self.__posterior_b_rho[i])
            return W_mu, W_rho, b_mu, b_rho

        def get_shape(self):
            return self.__shape_W, self.__shape_b

        def update(self, _q):
            self.assign(*_q.get_assignments())

        def train(self, observation, actions, targets):
            return self.__inference.update({
                self.__stateX: observation,
                self.__actionX: actions,
                self.__actionTargets: targets
            })

        def computeValue(self, observation, noise_W, noise_b):
            variables = { self.__observation: observation }
            for i in range(len(self.__noise_W.keys())):
                variables[self.__noise_W[i]] = noise_W[i]
                variables[self.__noise_b[i]] = noise_b[i]
            return self.__session.run(self.__Q_mu, feed_dict=variables)
    

    class NormalSampler:
        def __init__(self, shape_W, shape_b):
            assert(len(shape_W) == len(shape_b))
            self.__shape_W = shape_W
            self.__shape_b = shape_b

        def sample(self, number):
            noise_W, noise_b = {}, {}
            for _i in range(len(self.__shape_W)):
                noise_W[_i] = np.random.randn(*([number] + self.__shape_W[_i]))
                noise_b[_i] = np.random.randn(*([number] + self.__shape_b[_i]))
            return noise_W, noise_b


    def __init__(self, config, double=False, debug=False):
        self.__enableDebug = debug
        self.__config = config
        self.__doubleVDQN = double

    def __debug(self, msg: str, newlines: int = 0):
        if(self.__enableDebug):
            print("{0}VDQN: {1}".format(
                "".join(["\n" for i in range(newlines)]),
                msg
            ))

    def __generateAction(self, _q, state):
        noise_W, noise_b = self.__n.sample(1)
        _qValue = _q.computeValue(state[None], noise_W, noise_b)
        action = np.argmax(_qValue.flatten())
        return action
 

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
        rewardScaling = 1  # self.__config.get("reward_scaling")
        minibatchSize = self.__config.get("minibatch_size")
        hiddenLayers = [self.__config.get("hidden_layers")] * 2
        gamma = self.__config.get("gamma")
        tau = self.__config.get("tau")
        sigma = self.__config.get("sigma")
        networkUpdateFrequency = self.__config.get("network_update_frequency")
        maximumNumberOfSteps = self.__config.get("maximum_timesteps")

        # Initialise storage queues.
        replayBuffer = ReplayBuffer(capacity = 10**6)
        episodeTotals = deque(maxlen = self.__config.get("episode_history_averaging"))
        variationalLosses = deque(maxlen = self.__config.get("episode_history_averaging"))
        bellmanLosses = deque(maxlen = self.__config.get("episode_history_averaging"))

        with tf.Session() as session:
            _q = self.VariationalQFunction(obvSpace, actSpace, hiddenLayers, session, optimiser=tf.train.AdamOptimizer(self.__config.get("loss_rate")), scope="primary")
            _qTarget = self.VariationalQFunction(obvSpace, actSpace, hiddenLayers, session, optimiser=tf.train.AdamOptimizer(1e-3), scope="target")
            _n = self.NormalSampler(*_q.get_shape())
            self.__n = _n
            session.run(tf.global_variables_initializer())

            _qTarget.update(_q)

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

                    # Decay the epsilon value as the episode progresses.
                    epsilon = 0.6
                    if(len(replayBuffer) >= replayStartThreshold):
                        epsilon = max(
                            minimumEpsilon,
                            np.interp(
                                iteration,
                                [0, epsilonDecayPeriod],
                                [epsilon, minimumEpsilon]
                            )
                        )

                    # Select action to perform.
                    # Either random or greedy depending on the current epsilon value.
                    action = environment.action_space.sample() \
                        if np.random.rand() < epsilon \
                        else self.__generateAction(_q, currentState)

                    self.__debug("Episode {}: Timestep {}".format(episode, timestep))

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
                        minibatch = replayBuffer.randomSample(minibatchSize)
                        noise_W, noise_b = _n.sample(minibatchSize)

                        _alpha = _qTarget.computeValue(minibatch["nextStates"], noise_W, noise_b)
                        if self.__doubleVDQN:
                            _beta = _q.computeValue(minibatch["nextStates"], noise_W, noise_b)
                            _qNext = functions.select_item(_alpha, functions.argmax(_beta, axis=1))
                        else:
                            _qNext = functions.max(_alpha, axis=1)
                        
                        _qTargetValue = gamma * _qNext.array * (1 - minibatch["completes"]) + minibatch["rewards"]
                        _loss = _q.train(minibatch["states"], minibatch["actions"], _qTargetValue)
                        variationalLosses.append(_loss["loss"])

                        noise_W_dup, noise_b_dup = noise_W, noise_b
                        _prediction = _q.computeValue(minibatch["states"], noise_W_dup, noise_b_dup)
                        _predictedAction = _prediction[np.arange(minibatchSize), minibatch["actions"]]
                        _bellmanLoss = np.mean((_predictedAction - _qTargetValue)**2)
                        bellmanLosses.append(_bellmanLoss)

                    # Update the target Q network.
                    if iteration % networkUpdateFrequency == 0:
                        _qTarget.update(_q)

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
                    "variationalLosses": np.mean(variationalLosses),
                    "bellmanLosses": np.mean(bellmanLosses),                    
                })
