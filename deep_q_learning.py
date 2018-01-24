import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import time
import gym
import random
import matplotlib.pyplot as plt
import copy

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras import models

import keras.backend as K


def Q():
    model = Sequential()
    model.add(Dense(16, activation="relu", input_dim=4,
    kernel_regularizer=regularizers.l2(1e-5),activity_regularizer=regularizers.l2(1e-5)))
    model.add(Dense(2, activation='linear'))
    adam = optimizers.Adam(lr=1e-3)  # clipnorm=10
    model.compile(optimizer=adam,
                  loss='mse')
    return model


def learn(model,experiences, gamma):

    a = min(len(experiences),64)
    # sub_experiences=experiences
    # sub_experiences=random.sample([i for i in experiences if i[-1]==True]+random.sample(experiences,a),a)
    sub_experiences=random.sample(experiences,a)
    # sub_experiences = random.sample(experiences, a)

    x = np.array([i[0] for i in sub_experiences]).reshape(-1, 4)

    w = np.zeros((1, 4))

    y_ = model.predict(np.array(
        [i[-2] if i[-1] == False else w for i in sub_experiences]).reshape(-1, 4))

    y = model.predict(x)

    for i in range(len(sub_experiences)):
        action = sub_experiences[i][1]
        reward = sub_experiences[i][2]
        if sub_experiences[i][-1]:
            y[i][action] = reward
        else:
            # model.predict(sub_experiences[i][-1].reshape(1,4))
            y[i][action] = reward + gamma * np.amax(y_[i])

    model.fit(x, y, epochs=1, verbose=0, batch_size=2**8)


def display(model, env):
    observation = env.reset().reshape(1, -1)
    done=False
    while done!=True:
        env.render()
        prediction = model.predict(observation)[0]
        action = np.argmax(prediction)
        observation, reward, done, info = env.step(action)
        observation = observation.reshape(1, -1)


def main():
    gamma = 1

    lamb = 1e-3  # =0.46 a la 500eme iter

    # for choosing random action
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon = epsilon_start

    interval=1

    env = gym.make('CartPole-v0')
    model = Q()
    model_f=models.clone_model(model)
    reward_list = []
    q_list = []

    experiences = []

    for i_episode in range(1000):
        prev_observation = env.reset().reshape(1, -1)
        sum_reward = 0
        while True:
            # env.render()
            prediction = model.predict(prev_observation)[0]
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax(prediction)

            observation, reward, done, info = env.step(action)

            observation = observation.reshape((1, -1))
            sum_reward += reward
            if done:
                experiences.append(
                    [prev_observation, action, reward, None, done])
                break
            else:
                experiences.append(
                    [prev_observation, action, reward, observation, done])
                prev_observation=observation

                learn(model,experiences, gamma)

        # enable to not change the weights at each iterations for more stability (but I don't see any improvement by fixing it to 4 or 5)
        # if i_episode%interval==0:
        #     model_f=models.clone_model(model)

        reward_list.append(sum_reward)
        q_list.append(np.mean(prediction))

        epsilon = epsilon_end + \
            (epsilon_start - epsilon_end) * np.exp(-lamb * i_episode)


        if i_episode % 10 == 0:
            plt.yscale('log')
            plt.plot(reward_list)
            plt.plot(q_list)
            plt.show()
            plt.pause(0.0001)
    display(model, env)


plt.ion()
main()
# plt.show()
