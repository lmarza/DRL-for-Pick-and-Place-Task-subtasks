import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 20
LOG_SIG_MIN = -20

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(28, 128)
        self.fc2 = nn.Linear(128, 128) 
        self.lstm = nn.LSTMCell(128, 32)
        self.critic_linear = nn.Linear(32, 1)
        self.actor_linear = nn.Linear(32, 3)        
        self.apply(weights_init)
        
    def forward(self, x, hx, cx):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128)
        hx, cx = self.lstm(x, (hx, cx))
        x= hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

class second(nn.Module):
    def __init__(self):
        super(second, self).__init__()    
        self.fc1 = nn.Linear(28, 128)
        self.fc2 = nn.Linear(128, 128)    
        self.mean_linear = nn.Linear(128, 1)
        self.log_std_linear = nn.Linear(128, 1)
        self.mean_linear1 = nn.Linear(128, 1)
        self.log_std_linear1 = nn.Linear(128, 1)
        self.mean_linear2 = nn.Linear(128, 1)
        self.log_std_linear2 = nn.Linear(128, 1)
        self.mean_linear3 = nn.Linear(128, 1)
        self.log_std_linear3 = nn.Linear(128, 1)
        self.approachz = nn.Linear(128, 1)
        self.log_std_approachz = nn.Linear(128,1)
        self.approachy = nn.Linear(128, 1)
        self.log_std_approachy = nn.Linear(128,1)
        self.approachx = nn.Linear(128, 1)
        self.log_std_approachx = nn.Linear(128,1)
        self.retractz = nn.Linear(128, 1)
        self.log_std_retractz = nn.Linear(128,1)
        self.retracty = nn.Linear(128, 1)
        self.log_std_retracty = nn.Linear(128,1)
        self.retractx = nn.Linear(128, 1)
        self.log_std_retractx = nn.Linear(128,1)
        self.apply(weights_init)

    def forward(self, x, act):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        if act ==0 :    
            m0 = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            m1 = self.mean_linear1(x)
            log_std1 = self.log_std_linear1(x)
            log_std1= torch.clamp(log_std1, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            m2 = self.mean_linear2(x)
            log_std2 = self.log_std_linear2(x)
            log_std2= torch.clamp(log_std2, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            m3 = self.mean_linear3(x)
            log_std3 = self.log_std_linear3(x)
            log_std3= torch.clamp(log_std3, min=LOG_SIG_MIN, max=LOG_SIG_MAX)   
            list1 = torch.cat([m0, m1, m2, m3]).type(torch.cuda.FloatTensor)
            list2 = torch.cat([log_std, log_std1, log_std2, log_std3]).type(torch.cuda.FloatTensor)
            return list1, list2.exp()
        if act ==1:
            Az = self.approachz(x)
            log_stdz= self.log_std_approachz(x)
            log_stdz= torch.clamp(log_stdz, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            Ay = self.approachy(x)
            log_stdy= self.log_std_approachy(x)
            log_stdy= torch.clamp(log_stdy, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            Ax = self.approachx(x)
            log_stdx= self.log_std_approachx(x)
            log_stdx= torch.clamp(log_stdx, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            list1 = torch.cat([Az, Ay, Ax]).type(torch.cuda.FloatTensor)
            list2 = torch.cat([log_stdz, log_stdy, log_stdx]).type(torch.cuda.FloatTensor)
            return list1, list2.exp()
        if act == 2:
            Rz = self.retractz(x)
            log_stdRz= self.log_std_retractz(x)
            log_stdRz= torch.clamp(log_stdRz, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            Ry = self.retracty(x)
            log_stdRy= self.log_std_retracty(x)
            log_stdRy= torch.clamp(log_stdRy, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            Rx = self.retractx(x)
            log_stdRx= self.log_std_retractx(x)
            log_stdRx= torch.clamp(log_stdRx, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            list1 = torch.cat([Rz, Ry, Rx]).type(torch.cuda.FloatTensor)
            list2 = torch.cat([log_stdRz, log_stdRy, log_stdRx]).type(torch.cuda.FloatTensor)
            return list1, list2.exp()
        
def act(state_inp, act, model):
    FloatTensor = torch.cuda.FloatTensor
    list1, list2 = model(state_inp, act)
    #Az, Ay, Ax, m0, m1, m2, m3, Rz, Ry, Rx, log_stdz, log_stdy, log_stdx, log_std, log_std1, log_std2, log_std3, log_stdRz, log_stdRy, log_stdRx
    num = len(list1)
    act = torch.zeros(num).type(FloatTensor)
    for i in range(len(list1)):
        normal = Normal(list1[i], list2[i])
        X = normal.rsample()
        act[i] = torch.tanh(X)
        
    return act
    


class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params["action_max"]  # 1.0
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"], 256)  # 25 + 3 = 28, 256
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params["action"])  # 256, 6

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions
    