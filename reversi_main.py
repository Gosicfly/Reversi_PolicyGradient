import random
import gym
import numpy as np

from RL_QG_agent import RL_QG_agent

env = gym.make('Reversi8x8-v0')
env.reset()

agent = RL_QG_agent(train=False)

max_epochs = 100
model_win = 0

for i_episode in range(max_epochs):
    if (i_episode + 1) % 100 == 0:
        print('Episode {} done'.format(i_episode + 1))
    observation = env.reset()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    for t in range(100):
        action = [1,2]
        # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）
        ################### 黑棋 ############################### 0表示黑棋
        #  这部分 黑棋 是模型下棋
        # env.render()  #  打印当前棋局
        enables = list(set(env.possible_actions))
        if len(enables) == 0:
            action_ = env.board_size**2 + 1
        else:
            action_ = agent.place(observation, enables)  # 调用自己训练的模型
        action[0] = action_
        action[1] = 0   # 黑棋 为 0
        observation, reward, done, info = env.step(action)
        ################### 白棋 ############################### 1表示白棋
        # env.render()
        enables = list(set(env.possible_actions))
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1 # pass
        else:
            action_ = random.choice(enables)
        action[0] = action_
        action[1] = 1  # 白棋 为 1
        observation, reward, done, info = env.step(action)
        if done:
            agent.finish_episode(reward, agent.train)
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            if black_score >32:
                model_win += 1
            break

if agent.train:
    agent.save_model()

print('模型胜利次数：{}\t总次数：{}'.format(model_win, max_epochs))
print('模型胜率：{}'.format(model_win/max_epochs))