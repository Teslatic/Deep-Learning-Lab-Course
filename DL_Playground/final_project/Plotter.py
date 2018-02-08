import numpy as np
import matplotlib.pyplot as plt
from utils import Options
import seaborn as sns


class PlotterDude():

    # data is expected to be a list of lists
    def plot_avg_reward(data, agent_id):
        merged_mean = np.mean(data, axis = 0)
        std_data = np.std(data, axis = 0)
        plt.figure()
        for idx,val in enumerate(data):
            plt.plot(data[idx], label = 'Run {}'.format(idx))
        plt.plot(merged_mean, color = 'red', label='mean')
        plt.plot(merged_mean+std_data, color = 'pink', linestyle='--', label='mean + std. dev.')
        plt.plot(merged_mean-std_data, color = 'pink', linestyle='--', label='mean - std. dev.')
        plt.title('{}: Intermediate Test Run every {} Training Episodes [Avg. Reward]'.format(agent_id, Options.TEST_PROGRESS))
        plt.xlabel('Test [No.]')
        plt.ylabel('Reward (Vanilla)')
        plt.legend()

    def plot_avg_steps(data, agent_id):
        merged_mean = np.mean(data, axis = 0)
        std_data = np.std(data, axis = 0)
        plt.figure()
        for idx,val in enumerate(data):
            plt.plot(data[idx], label = 'Run {}'.format(idx))
        plt.plot(merged_mean, color = 'red', label='mean')
        plt.plot(merged_mean+std_data, color = 'pink', linestyle='--', label='mean + std. dev.')
        plt.plot(merged_mean-std_data, color = 'pink', linestyle='--', label='mean - std. dev.')
        plt.title('{}: Intermediate Test Run every {} Training Episodes [Steps needed]'.format(agent_id, Options.TEST_PROGRESS))
        plt.xlabel('Test [No.]')
        plt.ylabel('Steps needed')
        plt.legend()


    def plot_merged_means(data_DQN, data_PG, id):
        merged_mean_DQN = np.mean(data_DQN, axis = 0)
        merged_mean_PG = np.mean(data_PG, axis = 0)
        the_mean = np.mean([merged_mean_DQN, merged_mean_PG], axis = 0)
        the_std = np.std([merged_mean_DQN, merged_mean_PG], axis = 0)

        plt.figure()
        plt.plot(merged_mean_DQN, label='DQN')
        plt.plot(merged_mean_PG, label='PG')
        plt.plot(the_mean, label='mean')
        plt.plot(the_mean+the_std, color = 'pink', linestyle='--', label='mean + std. dev.')
        plt.plot(the_mean-the_std, color = 'pink', linestyle='--', label='mean - std. dev.')
        plt.title('Comparison of DQN and PG regarding [{}]'.format(id))
        plt.ylabel(id)
        plt.xlabel('Test [No.]')
        plt.legend()

    def policy_heatmap(data,  id):

        pos = []
        vel = []
        act = []

        for p,v,a in data:
            pos.append(p)
            vel.append(v)
            act.append(a)


        print(len(pos))
        print(len(vel))
        print(len(vel))


        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        x = np.linspace(0, len(pos),len(pos))
        y = np.linspace(0, len(vel),len(vel))

        def f(x, y, action):
            return x*action+y*action

        print(pos)
        print(vel)
        print(act)
        x = np.linspace(0, pos[-1],len(pos))
        y = np.linspace(0, vel[-1],len(vel))


        X, Y = np.meshgrid(x, y)
        Z = f(X, Y, act)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.show()


    def show_plots():
        plt.show()
