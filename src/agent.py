import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Distribution
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from typing import Any, Tuple
from torch import Tensor
from src.replay_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, num_macros_to_place: int, grid: int):
        super().__init__()

        self.grid = grid

        self.conv_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=int(grid / 128), stride=int(grid / 128)),
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4 * 32),
            nn.ReLU(),
        )

        self.up_5 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_5 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.up_6 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_6 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.up_7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_7 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_8 = nn.Sequential(
            nn.Upsample(scale_factor=int(grid / 128), mode='bilinear', align_corners=True),
            nn.Conv2d(8, 1, kernel_size=1, stride=1),
        )

        self.time_embedding = nn.Embedding(num_macros_to_place, 32)
    
    def __call__(self, s, t, *args: Any, **kwds: Any) -> Tuple[Distribution, Tensor, Tensor]:
        logits = self.forward(s, t)
        distr, pi_distr, action_distr, pi_distr_nomask = self.get_distr(logits, position_mask=kwds['position_mask'], wire_mask=kwds['wire_mask'])
        return distr, pi_distr, action_distr, pi_distr_nomask

    def forward(self, s, t):   # [B, 3, 256, 256]
        
        feature_1 = self.conv_1(s)  # [B, 8, 256, 256]
        feature_1_pool = self.pool_1(feature_1) # [B, 8, 64, 64]

        feature_2 = self.conv_2(feature_1_pool) # [B, 16, 128, 128]
        feature_2_pool = self.pool_2(feature_2) # [B, 16, 16, 16]

        feature_3 = self.conv_3(feature_2_pool) # [B, 32, 16, 16]
        feature_3_pool = self.pool_3(feature_3) # [B, 32, 4, 4]

        feature_4 = self.conv_4(feature_3_pool) # [B, 32, 4, 4]

        time_embedding = self.time_embedding(t) # [B, 32]
        feature = torch.cat([torch.reshape(feature_4, [-1, 4 * 4 * 32]), time_embedding], dim=-1) # [B, 544]
        feature = torch.reshape(self.fc(feature), [-1, 32, 4, 4])

        feature_5 = self.up_5(feature)    # [B, 32, 16, 16]
        feature_5 = self.conv_5(torch.cat((feature_3, feature_5), dim=1))   # [B, 16, 16, 16]

        feature_6 = self.up_6(feature_5)    # [B, 16, 64, 64]
        feature_6 = self.conv_6(torch.cat((feature_2, feature_6), dim=1))   # [B, 16, 64, 64]

        feature_7 = self.up_7(feature_6)   # [B, 8, 128, 128]
        feature_7 = self.conv_7(torch.cat((feature_1, feature_7), dim=1))   # [B, 8, 128, 128]

        feature_8 = self.conv_8(feature_7)   # [B, 1, 128, 128]
        
        return torch.reshape(feature_8, [-1, self.grid * self.grid])

    def get_distr(self,
        logits: Tensor,
        position_mask: Tensor, wire_mask: Tensor
    ):

        mask = torch.where(position_mask > 0.5, torch.inf, wire_mask).reshape([-1, self.grid * self.grid])
        wire_min = mask.min(dim=-1, keepdim=True)[0]
        greedy_mask = torch.where(mask == wire_min, 0.0, - 1e10).view([-1, self.grid * self.grid])

        action_mask = torch.reshape(position_mask >= 0.5, [-1, self.grid * self.grid])
        pi_distr = torch.where(action_mask, - 1e10, logits)
        greedy_distr = pi_distr + greedy_mask
        pi_distr = torch.softmax(pi_distr, dim=-1)
        
        pi_distr_no_mask = torch.softmax(logits, dim=-1)
        
        greedy_distr = torch.softmax(greedy_distr, dim=-1)

        action_distr = greedy_distr
        distr = Categorical(action_distr)

        return distr, pi_distr, action_distr, pi_distr_no_mask


class Critic(nn.Module):
    def __init__(self, num_macros_to_place, grid):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=int(grid / 128), stride=int(grid / 128)),
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 32 + 32, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
        )

        self.time_embedding = nn.Embedding(num_macros_to_place + 1, 32)

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

        self.time_value = nn.Embedding(num_macros_to_place + 1, 1)

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return self.forward(*args, **kwds)
    
    def forward(self, s: Tensor, t: Tensor):   # [B, 3, 128, 128]

        feature_1 = self.conv_1(s)  # [B, 8, 256, 256]
        feature_1_pool = self.pool_1(feature_1) # [B, 8, 64, 64]

        feature_2 = self.conv_2(feature_1_pool) # [B, 16, 128, 128]
        feature_2_pool = self.pool_2(feature_2) # [B, 16, 16, 16]

        feature_3 = self.conv_3(feature_2_pool) # [B, 32, 16, 16]
        feature_3_pool = self.pool_3(feature_3) # [B, 32, 4, 4]

        feature_4 = self.conv_4(feature_3_pool) # [B, 32, 4, 4]

        time_embedding = self.time_embedding(t) # [B, 32]
        feature = torch.cat([torch.reshape(feature_4, [-1, 4 * 4 * 32]), time_embedding], dim=-1) # [B, 544]

        feature = self.fc1(feature)

        value = self.fc2(feature) + self.time_value(t) # [B, 1]

        return value

class Agent:
    def __init__(self,
        max_grad_norm,
        clip_epsilon,
        entropy_coef,
        lr_actor,
        lr_critic,
        actor_lr_anneal_rate,
        critic_lr_anneal_rate,
        num_macros_to_place,
        grid,
        lamda,
        gamma=0.99,
    ):
        super().__init__()

        self.actor_net = Actor(num_macros_to_place, grid)
        self.critic_net = Critic(num_macros_to_place, grid)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr_actor)
        
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr_critic)
        
        self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, actor_lr_anneal_rate)
        self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, critic_lr_anneal_rate)
        
        self.training_step = 0
        self.lamda = lamda

        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.grid = grid

    
    def set_up(self, device, tb_writer: SummaryWriter):
        self.device = device
        self.actor_net.to(device)
        self.critic_net.to(device)
        gird_X, grid_Y = torch.meshgrid(torch.linspace(0, 1, self.grid+2)[1:-1], torch.linspace(0, 1, self.grid+2)[1:-1], indexing='xy')
        self.grid_points = torch.stack([gird_X, grid_Y], dim=-1).reshape(-1, 2).to(device)
        self.tb_writer = tb_writer
        self.actor_net.grid_points = self.grid_points
    
    @torch.no_grad()
    def select_action(self, state: Tensor, time_step:int):
        
        state = state.unsqueeze(0).to(self.device)
        canvas = state[:,0]
        wire_mask = state[:,1]
        position_mask = state[:,2]
        time_step = torch.LongTensor([time_step]).to(self.device)

        self.actor_net.eval()
        distr, pi_prob, a_prob, pi_prob_nomask = self.actor_net(state, time_step, position_mask=position_mask, wire_mask=wire_mask)
        action = distr.sample()
        action_log_prob = distr.log_prob(action.squeeze()) #a_prob.log().flatten()[action]

        return action.item(), action_log_prob.item()

    @torch.no_grad()
    def act_greedy(self, state: Tensor):
        wire_mask = state[1].float().to(self.device).unsqueeze(0)
        position_mask = state[2].int().to(self.device).unsqueeze(0)
        mask = torch.where(position_mask > 0.5, torch.inf, wire_mask).reshape(self.grid * self.grid)
        wire_min = mask.min()
        action_cand = torch.where(mask == wire_min)[0]
        action_idx = torch.randint(len(action_cand), (1,)).item()
        return action_cand[action_idx].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def update(self, replay_buffer: ReplayBuffer, num_update_epochs: int, batch_size: int):
        """Update the agent's actor and critic networks by sampling from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample experiences from.
            num_update_epochs (int): The number of epochs to use for updating the networks.
            batch_size (int): The batch size to use for updating the networks.
        """
        
        s, t, a, a_logp, r, s_, t_, done = replay_buffer.get_dataset(self.device)

        adv = []
        self.critic_net.eval()
        with torch.no_grad():
            v_target_list = []
            target = 0
            for re, d in zip(reversed(r.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                if d: target = 0
                target = re + self.gamma * target
                v_target_list.append(target)
            v_target_list.reverse()
            v_target = torch.tensor(v_target_list, dtype=torch.float).view(-1, 1).to(self.device)
            adv = v_target - self.critic_net(s, t)
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

            distr, pi_prob, a_prob, pi_prob_nomask = self.actor_net(s, t, position_mask=s[:,2], wire_mask=s[:,1])
            a_logp = distr.log_prob(a.squeeze()).unsqueeze(1)


        self.actor_net.train()
        self.critic_net.train()
        for _ in range(num_update_epochs):

            for idx in BatchSampler(SubsetRandomSampler(range(len(s))), batch_size, False):

                distr, pi_prob, a_prob,pi_prob_nomask = self.actor_net(s[idx], t[idx], position_mask=s[idx][:,2], wire_mask=s[idx][:,1])
                distr_entropy = torch.mean(distr.entropy())
                a_logp_new = distr.log_prob(a[idx].squeeze()).unsqueeze(1)

                ratios = torch.exp(torch.clamp(a_logp_new - a_logp[idx], max=7))
                surr1 = ratios * adv[idx]
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv[idx]
                actor_loss = - torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                v_s = self.critic_net(s[idx], t[idx])
                critic_loss = F.smooth_l1_loss(v_s, v_target[idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

                self.tb_writer.add_scalar('Loss/actor_loss', actor_loss, self.training_step)
                self.tb_writer.add_scalar('Loss/critic_loss', critic_loss.mean(), self.training_step)
                self.tb_writer.add_scalar('Loss/entropy', distr_entropy, self.training_step)
                self.tb_writer.add_scalar('Params/lr_actor', self.actor_lr_scheduler.get_last_lr()[0], self.training_step)
                self.tb_writer.add_scalar('Params/lr_critic', self.critic_lr_scheduler.get_last_lr()[0], self.training_step)
