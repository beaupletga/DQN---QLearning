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
from keras import optimizers
from keras.layers import LeakyReLU
from keras import models

import keras.backend as K

def tuned(y_true,y_pred):
    return K.mean(K.argmax(y_pred))


def policy():
    model = Sequential()
    model.add(Dense(64, activation="linear", input_dim=4))
    model.add(LeakyReLU())
    model.add(Dense(2, activation='softmax'))
    adam = optimizers.RMSprop(lr=1e-3)  # clipnorm=10
    model.compile(optimizer=adam,
                  loss="binary_crossentropy")
    return model


def learn(model,experiences, gamma):
    a = min(len(experiences),64)

    sub_experiences=random.sample(experiences,a)

    x = np.array([i[0] for i in sub_experiences]).reshape(-1, 4)
    # w = np.zeros((1, 4))
    #
    # y_ = model.predict(np.array(
    #     [i[-2] if i[-1] == False else w for i in sub_experiences]).reshape(-1, 4))

    a=[1,0]
    b=[0,1]


    y=np.array([[sub_experiences[i][2]*a if sub_experiences[i][1]==0 else sub_experiences[i][2]*b for i in range(0,len(sub_experiences))]).reshape(-1,2)
    # y = model.predict(x)

    # for i in range(len(sub_experiences)):
    #     action = sub_experiences[i][1]
    #     reward = sub_experiences[i][2]
    #     if sub_experiences[i][-1]:
    #         y[i][action] = reward
    #     else:
    #         # model.predict(sub_experiences[i][-1].reshape(1,4))
    #         y[i][action] = reward + gamma * np.amax(y_[i])

    model.fit(x, y, epochs=1, verbose=0, batch_size=2**14)


def display(model, env):
    observation = env.reset().reshape(1, -1)
    done=False
    while done!=True:
        env.render()
        prediction = model.predict(observation)[0]
        action = np.argmax(prediction)
        observation, reward, done, info = env.step(action)
        observation = observation.reshape(1, -1)


def ajouter_experience(experiences,experiences_tmp,gamma):
    for i in range(0,len(experiences_tmp)-1):
        experiences_tmp[i][2]+=sum(gamma**j*experiences_tmp[i][2] for j in range(i+1,len(experiences_tmp)))
    experiences=experiences+experiences_tmp

    return experiences


def main():
    gamma = 1

    lamb = 1e-3  # =0.46 a la 500eme iter
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon = epsilon_start

    interval=2

    env = gym.make('CartPole-v0')
    model = policy()
    reward_list = []

    experiences = []

    for i_episode in range(1000):
        prev_observation = env.reset().reshape(1, -1)
        sum_reward = 0

        experiences_tmp=[]
        while True:
            env.render()
            prediction = model.predict(prev_observation)[0]
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax(prediction)

            observation, reward, done, info = env.step(action)

            observation = observation.reshape((1, -1))
            sum_reward += reward
            if done:
                experiences_tmp.append(
                    [prev_observation, action, reward, None, done])
                experiences=ajouter_experience(experiences,experiences_tmp,gamma)
                break
            else:
                experiences_tmp.append(
                    [prev_observation, action, reward, observation, done])
                prev_observation=observation

            learn(model,experiences, gamma)

    #     reward_list.append(sum_reward)
    #     q_list.append(np.mean(prediction))
    #
    #     epsilon = epsilon_end + \
    #         (epsilon_start - epsilon_end) * np.exp(-lamb * i_episode)
    #
    #     if i_episode%interval==0:
    #         model_f=models.clone_model(model)
    #
    #     if i_episode % 10 == 0:
    #         plt.yscale('log')
    #         plt.plot(reward_list)
    #         plt.plot(q_list)
    #         plt.show()
    #         plt.pause(0.0001)
    # display(model, env)


# plt.ion()
main()
# plt.show()
