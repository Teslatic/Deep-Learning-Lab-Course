#!/usr/bin/python3

from DQNTraining import DQNTraining
from PGTraining import PGTraining
from Plotter import PlotterDude

### DQN Training
#
# DQNTrainer = DQNTraining()
# DQNTrainer.train_DQNAgent()


### Policy Training

PGTrainer = PGTraining()
PGTrainer.train_PGAgent()



### PlotterDude plots everything

# PlotterDude.plot_avg_reward(DQNTrainer.list_avg_reward, 'DQN')
# PlotterDude.plot_avg_steps(DQNTrainer.list_avg_steps, 'DQN')

# PlotterDude.plot_avg_reward(PGTrainer.list_avg_reward, 'PG')
# PlotterDude.plot_avg_steps(PGTrainer.list_avg_steps, 'PG')

# PlotterDude.plot_merged_means(DQNTrainer.list_avg_reward, PGTrainer.list_avg_reward, 'Reward')
# PlotterDude.plot_merged_means(DQNTrainer.list_avg_steps, PGTrainer.list_avg_steps, 'Steps')

PlotterDude.policy_heatmap(PGTrainer.heatmap_memory, 'PG')


PlotterDude.show_plots()
