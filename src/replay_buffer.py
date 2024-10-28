import torch

class ReplayBuffer:
    def __init__(self, capacity, grid):
        self.s = torch.zeros([capacity, 3, grid, grid])
        self.t = torch.zeros([capacity], dtype=torch.long)
        self.a = torch.zeros([capacity, 1], dtype=torch.long)
        self.a_logp = torch.zeros([capacity, 1])
        self.r = torch.zeros([capacity, 1])
        self.s_ = torch.zeros([capacity, 3, grid, grid])
        self.t_ = torch.zeros([capacity], dtype=torch.long)
        self.dw = torch.zeros([capacity, 1])
        self.done = torch.zeros([capacity, 1])
        self.capacity = capacity
        self.count = 0

    def store(self, s, t, a, a_logp, r, s_, t_, done):
        assert self.count <= self.capacity
        self.s[self.count] = s
        self.t[self.count] = t
        self.a[self.count] = a
        self.a_logp[self.count] = a_logp
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.t_[self.count] = t_
        self.done[self.count] = done
        self.count += 1

    def get_dataset(self, device):
        s = self.s[:self.count].to(device)
        t = self.t[:self.count].to(device)
        a = self.a[:self.count].to(device)
        a_logp = self.a_logp[:self.count].to(device)
        r = self.r[:self.count].to(device)
        s_ = self.s_[:self.count].to(device)
        t_ = self.t_[:self.count].to(device)
        done = self.done[:self.count].to(device)

        self.count = 0
        return s, t, a, a_logp, r, s_, t_, done