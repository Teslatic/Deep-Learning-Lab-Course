#!/usr/bin/env python3


import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options
from astar_simulator import Simulator

class astar_solver():
    def __init__(self,num_episodes):
        self.number_of_demo_episodes = num_episodes
        self.needed_steps = []
        self._execute_demo()        
    def _execute_demo(self):
        # 0. initialization
        opt = Options()
        sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

        # 1. demo control loop
        if opt.disp_on:
            win_all = None
            win_pob = None
        epi_step = 0    # #steps in current episode
        nepisodes = 0   # total #episodes executed

        for ep in range(self.number_of_demo_episodes):
            for step in range(100):
                if epi_step == 0:
                    state = sim.newGame(opt.tgt_y, opt.tgt_x)
                    nepisodes += 1
                else:
                    # will perform A* actions
                    # this is the part where your agent later
                    # would take its action
                    state = sim.step(None)

                    # instead you could also take a random action
                    # state = sim.step(randrange(opt.act_num))

                epi_step += 1

                if state.terminal or epi_step >= opt.early_stop:
                    # astar always succeeds
                    self.needed_steps.append(epi_step)
                    epi_step = 0

                if opt.disp_on:
                    if win_all is None:
                        plt.subplot(121)
                        win_all = plt.imshow(state.screen)
                        plt.subplot(122)
                        win_pob = plt.imshow(state.pob)
                    else:
                        win_all.set_data(state.screen)
                        win_pob.set_data(state.pob)
                    plt.pause(opt.disp_interval)
                    plt.draw()
