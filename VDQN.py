import tensorflow.compat.v1 as tf
import edward
import edward.models as edm
import gym

class VDQN:

    class VariationalQFunction:
        def __init__(self, stateSpace, actionSpace, hiddenLayers, session,
                     optimiser=tf.train.AdamOptimizer(1e-3), scope='alpha',
                     parameterConfig={}):
            
            self.__stateSpace = stateSpace
            self.__actionSpace = actionSpace
            self.__hiddenLayers = hiddenLayers
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
            pass

        def __forwardComputation(self):
            pass

        def __prior(self, W_sigma, b_sigma):
            with tf.variable_scope("prior"):
                nnLayers = [self.__stateSpace] + self.__hiddenLayers + [self.__actionSpace]
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
            pass

        def __inference(self):
            pass

        def __assignments(self):
            pass

        def main(self):
            pass


    def __init__(self, config, double=False, debug=False):
        self.__enableDebug = debug
        self.__config = config

    def run(self):
        self.__debug(self.__config.get("episodes"))

        # Build OpenAI Gym environment
        environment = gym.make(self.__config.get("environment"))
        if hasattr(environment, 'env'):
            environment = environment.env
        obvSpace = environment.observation_space.low.size
        actSpace = environment.action_space.n
        hiddenLayers = [100, 100]

        with tf.session() as session:
            VDQN.VariationalQFunction(obvSpace, actSpace, hiddenLayers, session)