"""manual keypress agent"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from rl.memory import SequentialMemory

from agents.agent_keras_rl_dqn import TrumpPolicy, memory_limit, window_length
from gym_env import env


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='Custom_Q1', load_model=None, env=None):
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

    def initiate_agent(self, nb_actions):
        """initiate a deep Q agent"""

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=env.observation_space))  # pylint: disable=no-member
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)  # pylint: disable=unused-variable
        policy = TrumpPolicy()  # pylint: disable=unused-variable

        nb_actions = env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])


    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action


    def action(self, action_space, observation, info):  # pylint: disable=no-self-use,unused-argument
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = (observation, info)  # not using the observation for random decision
        action = None

        # decide if explore or explot

        # forward

        # save to memory

        # backward
        # decide what to use for training
        # update model
        # save weights

        return action


    def train(self, env_name):
        """Train a model"""
        # initiate training loop
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                  write_images=False)

        self.dqn.fit(self.env, nb_max_start_steps=nb_max_start_steps, nb_steps=nb_steps, visualize=False, verbose=2,
                     start_step_policy=self.start_step_policy, callbacks=[tensorboard])

        # Save the architecture
        dqn_json = self.model.to_json()
        with open("dqn_{}_json.json".format(env_name), "w") as json_file:
            json.dump(dqn_json, json_file)

        # After training is done, we save the final weights.
        self.dqn.save_weights('dqn_{}_weights.h5'.format(env_name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=5, visualize=False)
