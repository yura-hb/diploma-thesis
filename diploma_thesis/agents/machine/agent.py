

from abc import ABCMeta, abstractmethod


class Agent:
    """
    The main responsibility of machine is to manage communication between 3 components:
        1. State Encoder: Encoding of shopfloor state into suitable format for the model
        2. Model:
            * Predicting the next action based on the encoded state from input in format
              (state, action, next_state, reward)
            * Performing training step based on the experience
        3. Maintaining Experience: Storing the experience in replay buffer
    """

    def __init__(self):
        pass

