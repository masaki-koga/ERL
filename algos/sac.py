import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update


class SAC(object):
    def __init__(self, args, model_constructor):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.writer = args.writer

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = model_constructor.make_model('Tri_Head_Q').to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = model_constructor.make_model('Tri_Head_Q').to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(1, args.action_dim)).cuda().item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            self.log_alpha.to(device=self.device)

        self.actor = model_constructor.make_model('Gaussian_FF').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.num_updates = 0


    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch):

        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # ----------------------------
        # 1. Critic (Q-function) Update
        # ----------------------------
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _ = self.actor.noisy_action(next_state_batch, return_only_action=False)
            qf1_next_target, qf2_next_target, _ = self.critic_target.forward(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * (min_qf_next_target) * (1 - done_batch)
            self.writer.add_scalar('next_q', next_q_value.mean().item())

        # Critic Forward
        qf1, qf2, _ = self.critic.forward(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        q_loss_total = qf1_loss + qf2_loss
        self.writer.add_scalar('q_loss', q_loss_total.mean().item() / 2.0)

        # Critic Backward & Step
        self.critic_optim.zero_grad()
        q_loss_total.backward()
        self.critic_optim.step()

        # ----------------------------
        # 2. Actor (Policy) Update
        # ----------------------------
        # Criticの重みをフリーズ（勾配計算対象外にする）
        for p in self.critic.parameters():
            p.requires_grad = False

        # Actor Forward (再計算: 最新のCriticに対してPolicyを評価するため)
        pi, log_pi, _, _, _ = self.actor.noisy_action(state_batch, return_only_action=False)
        self.writer.add_scalar('log_pi', log_pi.mean().item())

        qf1_pi, qf2_pi, _ = self.critic.forward(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        self.writer.add_scalar('policy_q', min_qf_pi.mean().item())

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.writer.add_scalar('policy_loss', policy_loss.mean().item())

        # Actor Backward & Step
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Criticの重みのフリーズ解除
        for p in self.critic.parameters():
            p.requires_grad = True

        # ----------------------------
        # 3. Soft Update
        # ----------------------------
        self.num_updates += 1
        soft_update(self.critic_target, self.critic, self.tau)

