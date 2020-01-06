from collections import deque
import random
import numpy as np

class ReplayBuffer:
    class Experience():
        def __init__(self, state, action, reward, completed, nextState):
            self.state = state
            self.action = action
            self.reward = reward
            self.completed = completed
            self.nextState = nextState


    def __init__(self, capacity : int = 100):
        self.__capacity = capacity
        self.__current = 0
        self.__queue = deque()

    def __len__(self):
        return self.__current

    def __getitem__(self, arg):
        return self.__queue[arg]

    def __normalise(self):
        while(self.__current > self.__capacity):
            self.__queue.popleft()
            self.__current -= 1

    def add(self, experience : Experience):
        self.__queue.append(experience)
        self.__current += 1
        self.__normalise()

    def randomSample(self, number : int):
        minibatch = random.sample(self.__queue, number)
        states = [experience.state for experience in minibatch]
        actions = [experience.action for experience in minibatch]
        rewards = [experience.reward for experience in minibatch]
        completes = [experience.completed for experience in minibatch]
        nextStates = [experience.nextState for experience in minibatch]

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "completes": np.array(completes),
            "nextStates": np.array(nextStates)
        }




