import torch
import logging
import math
import gym
from gym import spaces
from src.place_db import PlaceDB

class Environment(gym.Env):
    def __init__(self,
        benchmark,
        benchmark_dir,
        rank_mode,
        wire_mask_scale,
        reward_scale,
        num_macros_to_place,
        grid,
    ):

        placedb = PlaceDB(benchmark, benchmark_dir, rank_mode)
        self.placedb = placedb

        self.max_height = placedb.max_height
        self.max_width = placedb.max_width
        
        self.num_macros = placedb.node_cnt
        self.num_nets = placedb.net_cnt
        
        self.num_macros_to_place = num_macros_to_place
        self.grid = grid
        self.node_name_list = placedb.node_id_to_name
        
        self.action_space = spaces.Discrete(self.grid * self.grid)
        self.state = torch.Tensor()
        self.last_reward = 0

        self.ratio = self.placedb.max_height / self.grid
        self.size_x = [max(1, math.ceil(self.placedb.node_info[node_name]
                           ['x'] / self.ratio)) for node_name in self.node_name_list]
        self.size_y = [max(1, math.ceil(self.placedb.node_info[node_name]
                           ['y'] / self.ratio)) for node_name in self.node_name_list]

        self.wire_mask_scale = wire_mask_scale
        self.reward_scale = reward_scale

        self.macros = [self.placedb.node_info[self.node_name_list[i]]
                       for i in range(self.num_macros)]

    def reset(self):
        self.t = 0
        self.macro_pos = {}
        self.net_bound_info = {}

        canvas = torch.zeros((self.grid, self.grid))
        wire_mask = torch.zeros((self.grid, self.grid))
        next_x = max(1, math.ceil(self.macros[0]['x'] / self.ratio))
        next_y = max(1, math.ceil(self.macros[0]['y'] / self.ratio))
        position_mask = self.calc_position_mask(canvas, next_x, next_y)

        self.state = torch.cat([
            canvas.unsqueeze(0),
            wire_mask.unsqueeze(0),
            position_mask.unsqueeze(0),
        ], dim=0)

        return self.state

    def step(self, action):
        canvas, wire_mask, position_mask = self.state[0], self.state[1], self.state[2]

        pos_x = round(action // self.grid)
        pos_y = round(action % self.grid)
        size_x = self.size_x[self.t]
        size_y = self.size_y[self.t]

        # place the macro
        canvas[pos_x: pos_x + size_x, pos_y: pos_y + size_y] = 1.0
        canvas[pos_x: pos_x + size_x, pos_y] = 0.5
        if pos_y + size_y - 1 < self.grid:
            canvas[pos_x: pos_x + size_x, max(0, pos_y + size_y - 1)] = 0.5
        canvas[pos_x, pos_y: pos_y + size_y] = 0.5
        if pos_x + size_x - 1 < self.grid:
            canvas[max(0, pos_x + size_x-1), pos_y: pos_y + size_y] = 0.5
        self.macro_pos[self.node_name_list[self.t]] = (
            pos_x, pos_y, size_x, size_y)

        # calculate the reward
        hpwl_increment = wire_mask[pos_x, pos_y] * self.wire_mask_scale
        reward = - hpwl_increment / self.reward_scale

        if position_mask[pos_x][pos_y] == 1:
            reward = -1
            logging.info("INVALID ACTION.")

        node_name = self.node_name_list[self.t]
        for net_name in self.placedb.node_to_net_dict[node_name]:
            pin_x = round((pos_x * self.ratio + self.placedb.node_info[node_name]['x'] / 2 +
                           self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"]) / self.ratio)
            pin_y = round((pos_y * self.ratio + self.placedb.node_info[node_name]['y'] / 2 +
                           self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"]) / self.ratio)
            if net_name in self.net_bound_info:
                self.net_bound_info[net_name]['max_x'] = max(
                    pin_x, self.net_bound_info[net_name]['max_x'])
                self.net_bound_info[net_name]['min_x'] = min(
                    pin_x, self.net_bound_info[net_name]['min_x'])
                self.net_bound_info[net_name]['max_y'] = max(
                    pin_y, self.net_bound_info[net_name]['max_y'])
                self.net_bound_info[net_name]['min_y'] = min(
                    pin_y, self.net_bound_info[net_name]['min_y'])
            else:
                self.net_bound_info[net_name] = {}
                self.net_bound_info[net_name]['max_x'] = pin_x
                self.net_bound_info[net_name]['min_x'] = pin_x
                self.net_bound_info[net_name]['max_y'] = pin_y
                self.net_bound_info[net_name]['min_y'] = pin_y

        self.t += 1
        done = self.is_done()

        if self.t < self.num_macros:
            next_x = math.ceil(max(
                1, self.placedb.node_info[self.placedb.node_id_to_name[self.t]]['x']/self.ratio))
            next_y = math.ceil(max(
                1, self.placedb.node_info[self.placedb.node_id_to_name[self.t]]['y']/self.ratio))
            position_mask = self.calc_position_mask(canvas, next_x, next_y)
            wire_mask = self.calc_wiremask()
        else:
            next_x = 0
            next_y = 0
            position_mask = self.calc_position_mask(canvas, next_x, next_y)
            wire_mask = torch.zeros((self.grid, self.grid))

        next_state = torch.cat([
            canvas.unsqueeze(0),
            wire_mask.unsqueeze(0),
            position_mask.unsqueeze(0),
        ], dim=0).float()

        self.state = next_state
        info = {
            "delta_hpwl": hpwl_increment,
            "wire_mask": wire_mask,
            "position_mask": position_mask
        }

        return next_state, reward, done, info

    def calc_position_mask(self, canvas, next_x, next_y):
        mask = torch.zeros((self.grid, self.grid))
        for node_name in self.macro_pos:
            startx = max(0, self.macro_pos[node_name][0] - next_x + 1)
            starty = max(0, self.macro_pos[node_name][1] - next_y + 1)
            endx = min(self.macro_pos[node_name][0] +
                       self.macro_pos[node_name][2] - 1, self.grid - 1)
            endy = min(self.macro_pos[node_name][1] +
                       self.macro_pos[node_name][3] - 1, self.grid - 1)
            mask[startx: endx + 1, starty: endy + 1] = 1
        mask[self.grid - next_x + 1:, :] = 1
        mask[:, self.grid - next_y + 1:] = 1
        return mask

    def calc_wiremask(self):
        wire_mask = torch.zeros((self.grid, self.grid))
        node_name = self.placedb.node_id_to_name[self.t]

        for net_name in self.placedb.node_to_net_dict[node_name]:
            if net_name in self.net_bound_info:
                delta_pin_x = round((self.placedb.node_info[node_name]['x']/2 +
                                     self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"])/self.ratio)
                delta_pin_y = round((self.placedb.node_info[node_name]['y']/2 +
                                     self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"])/self.ratio)
                start_x = self.net_bound_info[net_name]['min_x'] - delta_pin_x
                end_x = self.net_bound_info[net_name]['max_x'] - delta_pin_x
                start_y = self.net_bound_info[net_name]['min_y'] - delta_pin_y
                end_y = self.net_bound_info[net_name]['max_y'] - delta_pin_y

                wire_mask_x = torch.arange(
                    self.grid, dtype=torch.float).unsqueeze(1).repeat(1, self.grid)
                wire_mask_x[:max(start_x, 0)] = start_x - \
                    wire_mask_x[:max(start_x, 0)]
                wire_mask_x[max(start_x, 0): max(end_x + 1, 0)] = 0.0
                wire_mask_x[max(end_x + 1, 0):] = wire_mask_x[max(end_x + 1, 0):] - end_x

                wire_mask_y = torch.arange(
                    self.grid, dtype=torch.float).unsqueeze(0).repeat(self.grid, 1)
                wire_mask_y[:, :max(start_y, 0)] = start_y - \
                    wire_mask_y[:, :max(start_y, 0)]
                wire_mask_y[:, max(start_y, 0): max(end_y + 1, 0)] = 0.0
                wire_mask_y[:, max(end_y + 1, 0):] = wire_mask_y[:,
                                                                 max(end_y + 1, 0):] - end_y

                wire_mask += wire_mask_x + wire_mask_y

        return wire_mask / self.wire_mask_scale

    def is_done(self):
        return self.t >= self.num_macros or self.t >= self.num_macros_to_place
