"""Player based on a trained neural network"""
# pylint: disable=wrong-import-order
import logging
import time

import numpy as np

from gym_env.env import Action

import tensorflow as tf
import json

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
# from keras.callbacks import TensorBoard

from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy

from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from rl.core import Processor
import pickle

import tensorflow.keras.backend as K

autoplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 0  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 500  # before training starts, should be higher than start steps
nb_steps = 250000
memory_limit = int(nb_steps/3)
enable_double_dqn = False

log = logging.getLogger(__name__)

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

        # if load_model:
        #     self.model = self.load_model(load_model)


    def initiate_agent(self, env, model_name=None, load_memory=None, load_model=None, load_optimizer=None, load_dqn=None, batch_size=500, learn_rate=1e-3):
        """initiate a deep Q agent"""
        # tf.compat.v1.disable_eager_execution()

        self.env = env

        nb_actions = self.env.action_space.n

        if load_model:
            pass
        #     self.model, trainable_model, target_model = self.load_model(load_model)
        #     print(self.model.history)

        else:
            pass

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=env.observation_space))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!            
        if load_memory:
            # print(load_memory)
            # exit()
            try:
                memory = self.load_memory(load_memory)

            except:
                pass

        else:
            memory = SequentialMemory(limit=memory_limit, window_length=window_length)

        self.batch_size = batch_size
        self.policy = CustomEpsGreedyQPolicy()
        self.policy.env = self.env

        self.test_policy = CustomEpsGreedyQPolicy()
        self.test_policy.eps = 0.05
        self.test_policy.env = self.env

        self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-4)

        nb_actions = env.action_space.n
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=self.policy, test_policy=self.test_policy,
                            processor=CustomProcessor(), batch_size=self.batch_size,
                            train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        
        # timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(model_name)
        # self.tensorboard = MyTensorBoard(log_dir='./Graph/{}'.format(timestr), player=self)
        self.dqn.compile(Adam(lr=learn_rate), metrics=['mae'])

        if load_model:
            self.load_model(load_model)
            # self.dqn.trainable_model = trainable_model
            # self.dqn.target_model = target_model

        # self.reduce_lr = ReduceLROnPlateau

        if load_optimizer:
            self.load_optimizer_weights(load_optimizer)

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        _ = observation
        legal_moves_limit = [move.value for move in self.env.info['legal_moves']]
        action = np.random.choice(legal_moves_limit)

        return action

    def train(self, env_name, batch_size=500, policy_epsilon=0.2):
        """Train a model"""
        # initiate training loop

        train_vars = {'batch_size': batch_size, 'policy_epsilon': policy_epsilon}
        
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                  write_images=False)
        self.dqn.fit(self.env, nb_max_start_steps=nb_max_start_steps, nb_steps=nb_steps, visualize=False, verbose=2, 
            start_step_policy=self.start_step_policy, callbacks=[tensorboard])

        self.policy.eps = policy_epsilon

        self.dqn.save_weights("dqn_{}_model.h5".format(env_name), overwrite=True)

        # Save memory
        pickle.dump( self.dqn.memory, open( "train_memory_{}.p".format(env_name), "wb" ) )

        # Save optimizer weights
        symbolic_weights = getattr(self.dqn.trainable_model.optimizer, 'weights')
        optim_weight_values = K.batch_get_value(symbolic_weights)
        pickle.dump(optim_weight_values, open( 'optimizer_weights_{}.p'.format(env_name), "wb" ) )


        # # Dump dqn
        # pickle.dump(self.dqn, open( "dqn_{}.p".format(env_name), "wb" ))


        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=5, visualize=False)

    def load_model(self, env_name):
        """Load a model"""

        # Load the architecture
        # with open('dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
        #     dqn_json = json.load(architecture_json)

        self.dqn.load_weights("dqn_{}_model.h5".format(env_name))
        # model = keras.models.load_model("dqn_{}_model.h5".format(env_name))
        # trainable_model = keras.models.load_model("dqn_{}_trainable_model.h5".format(env_name))
        # target_model = keras.models.load_model("dqn_{}_target_model.h5".format(env_name), overwrite=True)

        # return model, trainable_model, target_model

    def load_memory(self, model_name):
        memory = pickle.load( open( 'train_memory_{}.p'.format(model_name), "rb" ) )
        return memory

    def load_optimizer_weights(self, env_name):
        optim_weights = pickle.load( open('optimizer_weights_{}.p'.format(env_name), "rb"))
        self.dqn.trainable_model.optimizer.set_weights(optim_weights)


    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = CustomEpsGreedyQPolicy()

        class CustomProcessor(Processor):  # pylint: disable=redefined-outer-name
            """The agent and the environment"""

            def process_state_batch(self, batch):
                """
                Given a state batch, I want to remove the second dimension, because it's
                useless and prevents me from feeding the tensor into my CNN
                """
                return np.squeeze(batch, axis=1)

            def process_info(self, info):
                processed_info = info['player_data']
                if 'stack' in processed_info:
                    processed_info = {'x': 1}
                return processed_info

        nb_actions = self.env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])  # pylint: disable=no-member

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render)

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action



class MyTensorBoard(TensorBoard):
    def __init__(self, log_dir, player):
        self.player = player
        super().__init__(log_dir=log_dir, histogram_freq=0, 
            write_graph=True, write_images=False, profile_batch = 100000000)#

        self.log_dir = log_dir
        # self.file_writer.set_as_default()

    def on_epoch_begin(self, epoch, logs=None):
        with self.file_writer.as_default():
            tf.summary.scalar('batch_size', data=self.player.batch_size, step=epoch)
        # self.file_writer.flush()



    # def on_train_begin(self, logs=None):

    #     with self.file_writer.as_default():
    #         tf.summary.scalar('batch_size', data=self.player.batch_size, step=0)
    #         tf.summary.scalar('policy_epsilon', data=self.player.policy.eps, step=0)
    #     self.file_writer.flush()
    #     self.file_writer.close()

    #     return


class CustomEpsGreedyQPolicy(EpsGreedyQPolicy):
    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        legal_moves_limit = [move.value for move in self.env.info['legal_moves']]


        norm_q_values = [x / sum(q_values) for x in q_values]

        processed_q_values = q_values[:]
        
        # Set expectation of illegal moves to zero
        # These moves will never be selected for exploration
        for x in range(len(norm_q_values)):
            if x not in legal_moves_limit:
                processed_q_values[x] = np.min(q_values) - 1

        if np.random.uniform() < self.eps:
            action = np.random.choice(legal_moves_limit)

        else:
            action = np.argmax(processed_q_values)

        log.info(f"Chosen action by keras-rl {action} - processed_q_values: {processed_q_values}")
        return action



class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        legal_moves_limit = [move.value for move in self.env.info['legal_moves']]


        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))

        # Set expectation of illegal moves to zero
        # These moves will never be selected for exploration
        for x in range(len(q_values)):
            if x not in legal_moves_limit:
                exp_values[x] = 0.0

        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")
        return action


class CustomProcessor(Processor):
    """The agent and the environment"""

    def __init__(self):
        """initizlie properties"""
        self.legal_moves_limit = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into cnn"""
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # on arrays allowed it seems

    def process_action(self, action):
        # """Find nearest legal action"""
        # if 'legal_moves_limit' in self.__dict__ and self.legal_moves_limit is not None:
        #     self.legal_moves_limit = [move.value for move in self.legal_moves_limit]
        #     if action not in self.legal_moves_limit:
        #         for i in range(5):
        #             action += i
        #             if action in self.legal_moves_limit:
        #                 break
        #             action -= i * 2
        #             if action in self.legal_moves_limit:
        #                 break
        #             action += i


        return action
