class AlgorithmConfig:

    def __init__(self, initDict):
        self.__attributes = {
            "output_path": initDict.get("output_path", "/tmp/VDQN"),
            "post_episode": initDict.get("post_episode", lambda x: print(x)),
            "episodes": initDict.get("episodes", 100),
            "environment": initDict.get("environment", "CartPole-v1"),
            "loss_rate": initDict.get("loss_rate", 1e-2),
            "replay_start_threshold": initDict.get("replay_start_threshold", 500),
            "minimum_epsilon": initDict.get("minimum_epsilon", 0.01),
            "epsilon_decay_period": initDict.get("epsilon_decay_period", 5000),
            "reward_scaling": initDict.get("reward_scaling", 1e-2),
            "minibatch_size": initDict.get("minibatch_size", 64),
            "hidden_layers": initDict.get("hidden_layers", 100),
            "gamma": initDict.get("gamma", 0.99),
            "network_update_frequency": initDict.get("network_update_frequency", 100),
            "episode_history_averaging": initDict.get("episode_history_averaging", 50),
        }



    def get(self, key):
        return self.__attributes.get(key)

    
