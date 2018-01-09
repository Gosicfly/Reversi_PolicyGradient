import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Categorical
import os

gamma = 0.9

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(3 * 64, 2 * 64)
        self.affine2 = nn.Linear(2 * 64, 64)
        self.saved_log_probs = []
        self.step = 0

    def forward(self, observation, enables, train):
        if enables[0] == 65:
            return 65
        self.step += 1
        observation = observation.reshape(3*64)
        observation = Variable(torch.from_numpy(observation), volatile=(not train)).float()
        x = F.relu(self.affine1(observation))
        x = self.affine2(x)
        x = x[enables]
        probs = F.softmax(x, dim=-1)
        sampler = Categorical(probs)
        action = sampler.sample()
        self.saved_log_probs.append(sampler.log_prob(action))
        return enables[action.data[0]]

class RL_QG_agent:
    def __init__(self, train=True):
        self.train = train
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        self.model = PolicyNet()
        if train:
            self.model = PolicyNet()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.load_model()

    def place(self,state,enables):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        action = self.model(state, enables, self.train)
        return action

    def finish_episode(self, r, train):
        if not train:
            return
        R = 0
        policy_loss = []
        rewards = []
        for _ in range(self.model.step):
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        for log_prob, reward in zip(self.model.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.model.step = 0
        del self.model.saved_log_probs[:]

    def save_model(self):
        torch.save(self.model, 'model')
        print('Save model successfully!')

    def load_model(self):
        self.model = torch.load(self.model_dir)
        print('Load model successfully!')
