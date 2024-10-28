import torch
import pickle
import numpy as np
import os
import hydra
from omegaconf import DictConfig
import logging
import treelib
from tensorboardX import SummaryWriter
from utils import set_seed, set_device
from src import Agent, Environment, ReplayBuffer


class SolutionPool:
    def __init__(self, pool_size: int, alpha: float) -> None:
        self.pool_size = pool_size
        self.alpha = alpha

        self.best_hpwl = 1e9
        self.best_hpwl_max = 1e9

        self.sols = []
        self.tree = treelib.Tree()
        self.tree.create_node(
            tag="root",
            identifier=str([]),
            data={
                "visit_time": 0,
                "hpwl": 1e9,
                "hist_hpwl": {},
                "score": 0.0}
        )
        self.frontiers = [[]]
        self.frontiers_hist = [[]]

    def update_sol(self, frontier_id, solution, hpwl, placement):
        if hpwl < self.best_hpwl_max:
            sol_record = {
                "solution": solution,
                "placement": placement,
                "hpwl": hpwl,
            }
            self.sols.append(sol_record)
            self.sols = sorted(self.sols, key=lambda x: x["hpwl"])
            self.sols = self.sols[:self.pool_size]

            self.best_hpwl_max = self.sols[-1]["hpwl"]
            self.best_hpwl = self.sols[0]["hpwl"]

        for i in range(len(solution)):
            node = self.tree.get_node(str(solution[:i]))
            if node:
                node.data["visit_time"] += 1
                node.data["score"] = - node.data["hpwl"] + \
                    self.alpha * 1 / np.sqrt(node.data["visit_time"])
                if i < len(solution) - 1 and solution[:i+1] not in self.frontiers_hist:
                    node.data["hist_hpwl"][str(solution[i])] = hpwl.item() if str(
                        solution[i]) not in node.data["hist_hpwl"] else min(node.data["hist_hpwl"][str(solution[i])], hpwl.item())
                    node.data["hpwl"] = min(node.data["hpwl"], hpwl.item())
            if node is None:
                self.tree.create_node(
                    tag=str(solution[i-1]),
                    identifier=str(solution[:i]),
                    parent=str(solution[:i-1]),
                    data={
                        "visit_time": 1,
                        "hpwl": hpwl.item(),
                        "score": - hpwl.item() + self.alpha * 1 / np.sqrt(1),
                        "hist_hpwl": {str(solution[i]): hpwl.item()},
                    }
                )

    @property
    def get_solution(self):
        return self.sols[0]["placement"], self.sols[0]["solution"], self.sols[0]["hpwl"]

    @property
    def depth_min_max(self):
        depth = [len(x) for x in self.frontiers]
        return min(depth), max(depth)

    def update_frontiers(self):
        front_scores = [(self.tree.get_node(str(x)).data["score"], x)
                        for x in self.frontiers]
        front_scores = sorted(
            front_scores, key=lambda x: self.tree.get_node(str(x[1])).data["hpwl"])
        assert len(front_scores) <= self.pool_size
 
        self.frontiers_cand = []
        for front in self.frontiers:
            for chid in self.tree.children(str(front)):
                if eval(chid.identifier) not in self.frontiers and eval(chid.identifier) not in self.frontiers_cand:
                    self.frontiers_cand.append(eval(chid.identifier))
        cand_scores = [(self.tree.get_node(str(x)).data["score"], x)
                       for x in self.frontiers_cand]
        cand_scores = sorted(
            cand_scores, key=lambda x: self.tree.get_node(str(x[1])).data["hpwl"])
        while len(cand_scores) > 0 and (cand_scores[0][0] >= front_scores[-1][0] or len(front_scores) < self.pool_size):
            node = self.tree.get_node(str(cand_scores[0][1]))
            parent = self.tree.parent(node.identifier)
            if node.tag in parent.data["hist_hpwl"]:
                parent.data["hist_hpwl"].pop(node.tag)
            parent.data["hpwl"] = 1e9 if len(parent.data["hist_hpwl"].values(
            )) == 0 else min(parent.data["hist_hpwl"].values())
            parent.data["score"] = - parent.data["hpwl"] + \
                self.alpha * 1 / np.sqrt(parent.data["visit_time"])

            front_scores.append(cand_scores[0])
            front_scores = sorted(
                front_scores, key=lambda x: x[0], reverse=True)
            self.frontiers.append(cand_scores[0][1])
            self.frontiers_hist.append(cand_scores[0][1])
            if len(self.frontiers) > self.pool_size:
                self.frontiers.remove(front_scores[-1][1])
                front_scores = front_scores[:-1]
            self.frontiers_cand.remove(cand_scores[0][1])
            cand_scores = cand_scores[1:]

    def sample(self):
        frontier_id = np.random.choice(len(self.frontiers))
        return frontier_id, self.frontiers[frontier_id]


class Trainer:

    def __init__(self, num_loops, num_update_epochs, update_batch_size, num_episodes_in_loop, num_macros_to_place, solution_pool_size, alpha, update_frontiers_begin, update_frontiers_freq, model_dir, sol_dir, max_episode):
        self.num_loops = num_loops

        self.num_update_epochs = num_update_epochs
        self.update_batch_size = update_batch_size
        self.num_episodes_in_loop = num_episodes_in_loop
        self.num_macros_to_place = num_macros_to_place

        self.buffer_capacity = num_episodes_in_loop * num_macros_to_place
        self.solution_pool_size = solution_pool_size
        self.alpha = alpha

        self.update_frontiers_begin = update_frontiers_begin
        self.update_frontiers_freq = update_frontiers_freq
        self.solution_pool = SolutionPool(solution_pool_size, alpha)

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.sol_dir = sol_dir

    def setup_tb_writer(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def train(self, agent: Agent, env: Environment):

        logging.info(f"==================== Training ====================")

        global_episode = 0
        for loop in range(self.num_loops):
            logging.info(
                f"-------------------- Loop {loop} --------------------")
            replay_buffer = ReplayBuffer(self.buffer_capacity, env.grid)
            for episode in range(self.num_episodes_in_loop):
                hpwl, sol_id, solution, placement = self.run_episode(
                    env, agent, replay_buffer, global_episode)

                logging.info(f"Episode {global_episode}, HPWL = {hpwl}")
                self.solution_pool.update_sol(
                    sol_id, solution, hpwl, placement)

                self.tb_writer.add_scalar(
                    'episodes/hpwl', hpwl, global_episode)
                self.tb_writer.add_scalar(
                    "episodes/best_hpwl", self.solution_pool.best_hpwl, global_episode)
                self.tb_writer.add_scalar(
                    "episodes/best_hpwl_max", self.solution_pool.best_hpwl_max, global_episode)
                self.tb_writer.add_scalar(
                    "episodes/depth_min", self.solution_pool.depth_min_max[0], global_episode)
                self.tb_writer.add_scalar(
                    "episodes/depth_max", self.solution_pool.depth_min_max[1], global_episode)
                global_episode += 1

            agent.update(replay_buffer,
                         self.num_update_epochs,
                         self.update_batch_size)

            if global_episode >= self.update_frontiers_begin and loop % self.update_frontiers_freq == 0:
                self.solution_pool.update_frontiers()

            placement, sol, hpwl = self.solution_pool.get_solution

            logging.info(f"Current best HPWL = {hpwl}.")
            torch.save(agent.actor_net.state_dict(),
                       os.path.join(self.model_dir, "actor_net.pth"))
            torch.save(agent.critic_net.state_dict(),
                       os.path.join(self.model_dir, "critic_net.pth"))

            sol_dir = self.sol_dir
            os.makedirs(sol_dir, exist_ok=True)
            with open(os.path.join(sol_dir, f"best_placement_{hpwl}.pkl"), 'wb') as f:
                pickle.dump(placement, f)

    def run_episode(self, env: Environment, agent: Agent, replay_buffer: ReplayBuffer, global_episode: int):
        # initialize the environment
        t, hpwl, score = 0, 0, 0
        solution, placement = [], []
        s = env.reset()
        done = False

        # randomly select a frontier to start an episode from
        sid, sol = self.solution_pool.sample()

        while True:
            if t < len(sol):
                # select the action from the solution
                a = sol[t]
                a_logp = 0.0
            else:
                a, a_logp = agent.select_action(s, t, global_episode)

            s_, r, done, info = env.step(a)
            placement.append(a)
            hpwl += info["delta_hpwl"]

            if done and env.num_macros > self.num_macros_to_place:
                # if the episode is done, place the remaining macros greedily
                s_done, s__done, a_done = s, s_, a
                s = s_
                for i in range(env.num_macros - self.num_macros_to_place):
                    a = agent.act_greedy(s)
                    s_, r_, done, info = env.step(a)
                    placement.append(a)
                    hpwl += info["delta_hpwl"]
                    r += r_
                    if i == env.num_macros - self.num_macros_to_place - 1:
                        replay_buffer.store(s_done, t, a_done, a_logp, r, s__done, t + 1, 1.0)
                        solution.append(a_done)
                    s = s_
                score += r
                break
            else:
                if t >= len(sol):
                    replay_buffer.store(s, t, a, a_logp, r, s_, t + 1, 0.0)
                solution.append(a)
                score += r
                
                t += 1
            s = s_

        return hpwl * env.ratio / 1e5, sid, solution, placement
  
@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(config: DictConfig):
    set_seed(config.seed)
    device = set_device(config.cuda)
    tb_writer = SummaryWriter(config.tb_dir)

    agent: Agent = hydra.utils.instantiate(config.agent)
    agent.set_up(device, tb_writer)
    env: Environment = hydra.utils.instantiate(config.env)
    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    trainer.setup_tb_writer(tb_writer)

    trainer.train(agent, env)


if __name__ == '__main__':
    main()
