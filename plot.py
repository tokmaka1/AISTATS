import tikzplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_1D(X_sample, Y_sample, X_plot, fX, title, safety_threshold, save):
    plt.figure()
    plt.plot(X_plot, fX, color='blue')
    plt.scatter(X_sample[1:], Y_sample[1:], color='black')
    plt.plot(X_plot, [safety_threshold]*len(X_plot), '-r')
    plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
    plt.xlabel('$a$')
    plt.ylabel('$y$')
    plt.title(title)
    if save:
        tikzplotlib.save(f'Experiments/1D_toy_experiments/{title}.tex')
        plt.savefig(f'Experiments/1D_toy_experiments/{title}.png')
    if not save:
        plt.legend(['Ground truth', 'Samples', 'Threshold'])
        plt.show()


def plot_2D_scatter(X_plot, fX, X_sample, safety_threshold):
    plt.figure()
    sc = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=fX, vmin=min(fX), vmax=max(fX), s=15, cmap=cm.jet)
    plt.colorbar(sc)
    plt.scatter(X_sample[:, 0], X_sample[:, 1], color='black', s=50)
    plt.plot(X_sample[0, 0], X_sample[0, 1], '*', color='magenta', markersize=10)
    plt.scatter(X_plot[:, 0][fX < safety_threshold], X_plot[:, 1][fX < safety_threshold], color='white', s=50)
    plt.title('Ground truth')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')


def plot_2D_contour(X_plot, fX, X_sample, Y_sample, safety_threshold, title, levels=10, save=False):
    division_points = int(np.sqrt(len(fX)))
    x1 = X_plot[:, 0].reshape(division_points, division_points)
    x2 = X_plot[:, 1].reshape(division_points, division_points)

    # Define contour levels

    # Create contour plot using Matplotlib
    plt.figure()
    contour = plt.contour(x1, x2, fX.reshape(division_points, division_points), levels=levels, cmap='seismic')
    plt.colorbar(contour)
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    plt.title(f'{title} threshold: {safety_threshold}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    # Color the contours based on their value
    plt.scatter(X_sample[1:, 0], X_sample[1:, 1], color='black', s=50)
    plt.scatter(X_sample[0, 0], X_sample[0, 1], color='magenta', s=50)
    for i in range(len(Y_sample)):
        if Y_sample[i] < safety_threshold:
            plt.scatter(X_sample[i, 0], X_sample[i, 1], marker='s', color='red', s=150)  # Corrected this line
    if save:
        tikz_code = tikzplotlib.get_tikz_code(float_format=".5g")
        with open(f'Experiments/2D_toy_experiments/{title}.tex', 'w') as f:
            f.write(tikz_code)
        plt.savefig(f'Experiments/2D_toy_experiments/{title}.png')
    else:
        plt.show()


def plot_1D_SafeOpt_with_sets(global_cube, gt, save=False, title=None):
    X_plot = gt.X_plot
    plt.figure()
    plt.plot(X_plot, [gt.safety_threshold]*len(X_plot), '-r')
    plt.scatter(global_cube.x_sample, global_cube.y_sample, color='k')
    plt.plot(X_plot, gt.fX, '-b')
    plt.plot(X_plot, global_cube.mean, '-k')
    plt.fill_between(X_plot.flatten(), global_cube.lcb, global_cube.ucb, alpha=0.1, color='gray')
    min_lcb = min(global_cube.lcb)
    for x in X_plot[global_cube.M]:
        plt.plot(x, min_lcb-0.5, marker='s', markersize=5, color='orange')
    for x in X_plot[global_cube.G]:
        plt.plot(x, min_lcb-1, marker='s', markersize=5, color='purple')
    for x in X_plot[global_cube.S]:
        plt.plot(x, min_lcb, marker='s', markersize=5, color='cyan')
    if save:
        plt.title = title
        tikz_code = tikzplotlib.get_tikz_code(float_format=".5g")
        with open(f'Experiments/SafeOpt_RKHS_intro/{title}.tex', 'w') as f:
            f.write(tikz_code)
        plt.savefig(f'Experiments/SafeOpt_RKHS_intro/{title}.png')
    else:
        plt.show()


def plot_gym(Y_sample, safety_threshold=0, title=None, save=False):
    plt.figure()
    plt.plot(range(len(Y_sample), Y_sample))
    plt.plot(range(len(Y_sample)), [safety_threshold]*len(Y_sample), 'red')
    if title is not None:
        plt.title(f'{title}')
    if save:
        plt.savefig(f'{title}.png')
        tikzplotlib.save(f'{title}.tex')
    else:
        plt.show()

def plot_gym_together(Y_sample_SO_under, Y_sample_SO_over, Y_sample_pbo, safety_threshold=0, title=None, save=False):
    plt.figure()
    plt.plot(range(len(Y_sample_SO_under)), Y_sample_SO_under, label='SafeOpt under')
    plt.plot(range(len(Y_sample_SO_over)), Y_sample_SO_over, label='SafeOpt over')
    plt.plot(range(len(Y_sample_pbo)), Y_sample_pbo, label='PACSBO')
    max_len = max(len(Y_sample_SO_over), len(Y_sample_SO_under), len(Y_sample_pbo))
    plt.plot(range(max_len), [safety_threshold]*max_len, 'red')
    plt.legend()
    if save and title is not None:
        plt.savefig(f'{title}.png')
        tikzplotlib.save(f'{title}.tex')
    else:
        plt.show()
