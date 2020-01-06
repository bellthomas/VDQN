import tensorflow as tf
import edward
import edward.models as edm
import gym
import numpy as np

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
            self.__sigmaRho = parameterConfig.get("sigmaRho", None) # Wpriorsigma=None, bpriorsigma=None
            
            _prior_W_sigma = parameterConfig.get("prior_W_sigma", None)
            _prior_b_sigma = parameterConfig.get("prior_b_sigma", None)
            with tf.variable_scope(scope):
                self.__prior(_prior_W_sigma, _prior_b_sigma)
                self.__model()
                self.__posterior()
                self.__forwardComputation()
                self.__inference()
                self.__assignments()


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
            self.__inference.initialise(
                optimizer = self.__optimiser,
                scale = { self.__nextAction: 1 },
                n_iter = iterations
            )

        def __assignments(self):
            self.__values_W_mu = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_W_mu.items() }
            self.__values_W_rho = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_W_rho.items() }
            self.__values_b_mu = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_b_mu.items() }
            self.__values_b_rho = { x: tf.placeholder(tf.float32, y.shape) for x, y in self.__posterior_b_rho.items() }
            self.__assignments = []

            layers = len(self.__values_W_mu.keys())
            _t = self.__tau
            for _i in range(layers):
                _alpha = self.__posterior_W_mu[_i].assign(
                    _t * self.__values_W_mu[_i] + (1-_t) * self.__posterior_W_mu[_i]
                )
                _beta = self.__posterior_W_rho[_i].assign(
                    _t * self.__values_W_rho[_i] + (1-_t) * self.__posterior_W_rho[_i]
                )
                _gamma = self.__posterior_b_mu[_i].assign(
                    _t * self.__values_b_mu[_i] + (1-_t) * self.__posterior_b_mu[_i]
                )
                _delta = self.__posterior_b_rho[_i].assign(
                    _t * self.__values_b_rho[_i] + (1-_t) * self.__posterior_b_rho[_i]
                )
                self.__assignments.extend([_alpha, _beta, _gamma, _delta])

        # def assign(self):

        def train(self, observation, actions, targets):
            return self.__inference({
                self.__stateX: observation,
                self.__actionX: actions,
                self.__actionTargets: targets
            })
    
        def main(self):
            pass


    def __init__(self, config, double=False, debug=False):
        self.__enableDebug = debug
        self.__config = config

    def __debug(self, msg: str, newlines: int = 0):
        if(self.__enableDebug):
            print("{0}VDQN: {1}".format(
                "".join(["\n" for i in range(newlines)]),
                msg
            ))

    def run(self):
        self.__debug(self.__config.get("episodes"))

        # Build OpenAI Gym environment
        environment = gym.make(self.__config.get("environment"))
        if hasattr(environment, 'env'):
            environment = environment.env
        obvSpace = environment.observation_space.low.size
        actSpace = environment.action_space.n
        hiddenLayers = [100, 100]

        with tf.Session() as session:
            VDQN.VariationalQFunction(obvSpace, actSpace, hiddenLayers, session)