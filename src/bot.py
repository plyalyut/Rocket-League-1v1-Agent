import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.orientation import Orientation
from util.vec import Vec3
import numpy as np
import os

class MyBot(BaseAgent):

    def initialize_agent(self):
        '''
        Initializes the agent and sets up all hyperparamters
        :return: No return
        '''

        # Sets up the controller agent
        self.controller_state = SimpleControllerState()

        # Initializes all value lists
        self.reset_episode_data()

        # Sets up all the hyperparameters
        self.episode_length = 10

        # Creates the model
        import tensorflow as tf
        self.tf = tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        class A2C_Continuous(self.tf.keras.Model):
            def __init__(self):
                super(A2C_Continuous, self).__init__()
                # Actor Network

                self.hidden_sz = 100
                self.num_actions = 2
                self.critic_scale = 0.5

                self.act1 = tf.keras.layers.Dense(units=self.hidden_sz, input_shape=(4,), activation="relu")
                self.act2 = tf.keras.layers.Dense(units=self.hidden_sz, input_shape=(4,), activation='relu')
                self.mean = tf.keras.layers.Dense(self.num_actions, activation="tanh")
                self.std = tf.keras.layers.Dense(self.num_actions, activation='sigmoid')

                # Critic Network
                self.crit1 = tf.keras.layers.Dense(units= self.hidden_sz, activation="relu")
                self.hidden_crit = tf.keras.layers.Dense(units = self.hidden_sz/2, activation = 'relu')
                self.hidden_crit2 = tf.keras.layers.Dense(units = self.hidden_sz/10, activation = 'relu')
                self.crit2 = tf.keras.layers.Dense(1)

                # Create optimizer
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            @tf.function
            def call(self, states):
                '''
                Computes the policies from the states
                :param states: np.array of (batch_size x num_states)
                :return: policies for the states of size (batch_size, num_actions)
                '''
                mean = self.mean(self.act1(states))
                std = self.std(self.act2(states))

                return mean, std

            @tf.function
            def critic(self, states):
                '''
                Computes the value at each of the states
                :param states: np.array of (batch size x num_states)
                :return: values at each of the states (batch_size, 1)
                '''

                return self.crit2( self.hidden_crit2(self.hidden_crit(self.crit1(states))))

            @tf.function
            def loss(self, states, actions, discounted_rewards):
                '''
                Computes the loss for a given reward
                :param states: a list of states (episode_length, num_states)
                :parma actions: all the actions that were taken in the episode (episode_length, num_actions)
                :param discounted_rewards: A list of discounted rewards (episode_length, 1)
                :return: Loss of both the actor and critic
                '''

                advantage = tf.cast(tf.cast(tf.reshape(discounted_rewards, (-1, 1)), dtype=tf.float32) - self.critic(states), dtype=tf.float64)

                mean, std = self.call(states)

                mean = tf.cast(mean, dtype=tf.float64)
                std = tf.cast(std, dtype=tf.float64)

                actions = tf.squeeze(actions)

                # Continuous A2C model
                pdf = tf.divide(1, tf.math.sqrt(2. * np.pi * tf.square(std))) * tf.exp(
                    -tf.divide(tf.square((actions - tf.cast(mean, dtype=tf.float64))), (2. * tf.square(std))))
                log_pdf = tf.math.log(pdf + 0.0000001)

                actor_loss = -tf.reduce_mean(log_pdf * tf.stop_gradient(advantage))
                critic_loss = -tf.reduce_mean(tf.square(advantage))

                return actor_loss + self.critic_scale * critic_loss

        class A2C(self.tf.keras.Model):
            def __init__(self):
                super(A2C, self).__init__()
                # Actor Network

                self.hidden_sz = 32
                self.num_actions = 4
                self.critic_scale = 0.5

                self.act1 = tf.keras.layers.Dense(units=self.hidden_sz, input_shape=(2,), activation="relu")
                self.act2 = tf.keras.layers.Dense(self.num_actions, activation='softmax')

                # Critic Network
                self.crit1 = tf.keras.layers.Dense(units= self.hidden_sz, activation="relu")
                self.crit2 = tf.keras.layers.Dense(1)

                # Create optimizer
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            @tf.function
            def call(self, states):
                '''
                Computes the policies from the states
                :param states: np.array of (batch_size x num_states)
                :return: policies for the states of size (batch_size, num_actions)
                '''

                return self.act2(self.act1(states))


            @tf.function
            def critic(self, states):
                '''
                Computes the value at each of the states
                :param states: np.array of (batch size x num_states)
                :return: values at each of the states (batch_size, 1)
                '''

                return self.crit2(self.crit1(states))

            def loss(self, states, actions, discounted_rewards):
                '''
                Computes the loss for a given reward
                :param states: a list of states (episode_length, num_states)
                :parma actions: all the actions that were taken in the episode (episode_length, num_actions)
                :param discounted_rewards: A list of discounted rewards (episode_length, 1)
                :return: Loss of both the actor and critic
                '''

                state_values = self.critic(states)
                policy = self.call(states)

                advantage = tf.cast(discounted_rewards - state_values, dtype=tf.float32)

                # The gather_nd is to get the probability of each action that was actually taken in the episode

                log_P = tf.math.log(tf.gather_nd(policy, list(zip(np.arange(len(policy)), actions))))

                actor_loss = -tf.reduce_sum(log_P * tf.stop_gradient(advantage))
                critic_loss = tf.reduce_sum(tf.square(advantage))
                return actor_loss + .5 * critic_loss

        self.a2c = A2C()

        print('Model Successfully Initialized Without any Issue')


    def get_reward(self, ball_location, goal_location, player_location):
        '''
        Reward function for the given play. In this case,
        the reward function is just the inverse distance between the
        goal and the ball. This means that the reward is higher
        whenever the ball is closer to the goal.
        :param ball_location: Vec3 location of the ball
        :param goal_location: Vec3 location of the goal
        :return: Reward, float
        '''

        #return 1.0/(1+(ball_location-goal_location).length()) + 1/(1+(player_location-goal_location).length())
        return 1/(1+(player_location-goal_location).length())

    def get_discounted_rewards(self, reward_list, discount_factor):
        '''
        Computes the discounted rewards for the episode
        :param reward_list: list of rewards for the entire play
        :return: List of discounted rewards
        '''
        prev = 0
        discounted_rewards = np.copy(reward_list).astype(np.float32)
        for i in range(1, len(discounted_rewards) + 1):
            discounted_rewards[-i] += prev * discount_factor
            prev = discounted_rewards[-i]

        return discounted_rewards

    def reset_episode_data(self):
        '''
        Creates new lists that would store all training values.
        :return: No return
        '''
        self.states = []
        self.actions = []
        self.rewards = []

    def convert_v3(self, vec3):
        '''
        Converts vec3 to a list
        :param vec3: Vector representaiton
        :return: list of x, y, z coordinates of a vector
        '''
        return [vec3.x/4096.0, vec3.y/5120.0]

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        '''
        Runs the model and captures the states, rewards, actions taken for the episode.
        :param packet: GameTickPacket contains all information about boost and such
        :return:
        '''

        # Gets the agent
        agent = packet.game_cars[self.index]

        # Gets the ball location as a vector
        ball_location = Vec3(packet.game_ball.physics.location)
        agent_location = Vec3(agent.physics.location)
        target_goal_location = self.get_field_info().goals[1-agent.team].location

        # State of the ball and car
        state = []
        #state.extend(self.convert_v3(agent_location-target_goal_location))
        state.extend(self.convert_v3(agent_location))

        # Generate the action it should take
        prob_action = self.a2c.call(np.reshape(np.array(state), [1, 2]))

        action = np.random.choice(4, p=np.squeeze(prob_action))
        print(action)

        # Sets all the controller states
        if action%2:
            self.controller_state.throttle = 1
        else:
            self.controller_state.throttle = 0
        if int(action/2) == 0:
            self.controller_state.steer = 1
        else:
            self.controller_state.steer = -1

        #self.controller_state.throttle = np.clip(action[0][0], -1, 1)
        #self.controller_state.steer = np.clip(action[0][1], -1, 1)

        # Keep track of all usable information
        self.states.append(state)
        self.rewards.append(self.get_reward(ball_location, target_goal_location, agent_location))
        self.actions.append(action)

        if len(self.states) >= self.episode_length:
            print('Training')

            self.states_copy = np.array(self.states)
            self.rewards_copy = np.array(self.rewards)
            self.actions_copy = np.array(self.actions)

            with self.tf.GradientTape() as tape:
                discounted_rewards = self.get_discounted_rewards(self.rewards_copy, 0.99)
                loss = self.a2c.loss(self.states_copy, self.actions_copy, discounted_rewards)
            draw_debug(self.renderer, np.sum(discounted_rewards), loss)
            gradients = tape.gradient(loss, self.a2c.trainable_variables)
            self.a2c.optimizer.apply_gradients(zip(gradients, self.a2c.trainable_variables))
            self.reset_episode_data()

        return self.controller_state


def draw_debug(renderer, reward, loss):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    # print the action that the bot is taking
    renderer.draw_string_2d(0, 0, 2, 2, 'Reward: ' + str(reward), renderer.white())
    renderer.draw_string_2d(0, 30, 2, 2, 'Loss:  ' + str(loss), renderer.white())
    renderer.end_rendering()




