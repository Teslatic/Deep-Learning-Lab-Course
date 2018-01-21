#!/usr/bin/env python3


import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options
from astar_simulator import Simulator

class astar_solver():
    def __init__(self,episodes):
        self.nepisodes = episodes
        self.list_epi_steps = []
        
        self._execute_demo()
        
    def _execute_demo(self):
                
            
        # 0. initialization
        opt = Options()
        self.sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
        terminal = False
        # 1. demo control loop
        opt.disp_on = False
        if opt.disp_on:
            win_all = None
            win_pob = None
        epi_step = 0    # #steps in current episode
        nepisodes = 0   # total #episodes executed

        for i in range(self.nepisodes):
            terminal = False
            while not terminal:
                if epi_step == 0:
                    state = self.sim.newGame(opt.tgt_y, opt.tgt_x)
                    nepisodes += 1
                else:
                    # will perform A* actions
                    # this is the part where your agent later
                    # would take its action
                    state = self.sim.step(None)
                    # instead you could also take a random action
                    # state = sim.step(randrange(opt.act_num))

                epi_step += 1

                if state.terminal or epi_step >= opt.early_stop:
                    # astar always succeeds optimal
                    self.list_epi_steps.append(epi_step)
                    epi_step = 0
                
                terminal = True if state.terminal else False

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
        
