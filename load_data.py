# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x_index = 7


def load_data():
    mat = h5py.File('./data/qian-data.mat')
    data = [np.transpose(mat[h5py.h5r.get_name(mat['BB'][i][j], mat.id)]) for i in range(len(mat['BB'])) for j in
            range(len(mat['BB'][0]))]
    return data


def visualize_trace(data):
    for i, line in enumerate(data):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(xs=line[x_index], ys=line[x_index + 1], zs=line[x_index + 2])
        ax.scatter(line[x_index][:1], line[x_index + 1][:1], line[x_index + 2][:1], 'r')
        fig.show()
        # fig.savefig(f"./figure/trace_{i}.png", dpi=300)


def visualize_point(data):
    for i, line in enumerate(data):
        fig = plt.figure()
        plt.scatter(line[x_index], line[x_index + 1])
        fig.show()
        # fig.savefig(f"./figure/trace_{i}.png", dpi=300)


def visualize_2D_trace(data):
    for i, line in enumerate(data):
        fig = plt.figure()
        plt.plot(get_xy_z(line, line[:, :1]), line[x_index + 2])
        fig.show()
        # fig.savefig(f"./figure/trace_{i}.png", dpi=300)


def get_xy_z(single_line_data, start_point):
    return ((single_line_data[x_index] - start_point[x_index][0]) ** 2 + (
                single_line_data[x_index + 1] - start_point[x_index + 1][0]) ** 2) ** 0.5


def get_curve_function(line_data, deg, start_point):
    param = np.polyfit(get_xy_z(line_data, start_point), line_data[x_index + 2], deg)
    # print(param)
    curve_function = np.poly1d(param)
    # print(curve_function)
    return param, curve_function

def compare_fit_curve(real_curve, fitted_curve_list):
    fig = plt.figure()
    color_list = ['r', 'g', 'gray']
    plt.plot(real_curve[0], real_curve[1], 'b')
    for i, fitted_curve in enumerate(fitted_curve_list):
        x, y = fitted_curve
        plt.plot(list(x), list(y), color_list[i % len(color_list)])
    fig.show()

def cluster_three(data_list):
    kmeans = KMeans(
        n_clusters=n_cluster,
        n_init=10,
        max_iter=3000,
        algorithm='auto').fit(data_list)
    return kmeans.cluster_centers_


# %%
if __name__ == '__main__':
    # %%
    trace_data = load_data()
    # %%
    visualize_trace(trace_data)
    # %%
    visualize_point(trace_data)
    # %%
    visualize_2D_trace(trace_data)
    # %%
    get_curve_function(trace_data[0], 3, trace_data[0][:, :1])
    # %%
    n_cluster = 3
    fit_step_length = 300
    allowed_diff = 1000
    deg = 3
    for i, single_data in enumerate(trace_data[:1]):
        param_list = []
        for j in range(single_data.shape[1] - (fit_step_length - 1)):
            example_data = single_data[:, j: j + fit_step_length]
            param, fit_function_ = get_curve_function(example_data, deg, single_data[:, :1])
            param_list.append(param)

        fit_function_list = [np.poly1d(param) for param in param_list]
        cluster_param_center = cluster_three(param_list)
        cluster_fit_function_list = [np.poly1d(param) for param in cluster_param_center]
        xy = get_xy_z(single_data, single_data[:, :1])

        fit_curve_list = []
        for fit_function in cluster_fit_function_list:
            curve_tuple = ([], [])
            enter = False
            for p_index, p_xy in enumerate(xy):
                fit_z = fit_function(p_xy)
                if abs(fit_z - single_data[x_index + 2][p_index]) < allowed_diff:
                    enter = True
                    curve_tuple[0].append(p_xy)
                    curve_tuple[1].append(fit_z)
                elif enter:
                    break
            fit_curve_list.append(curve_tuple)
        compare_fit_curve((xy, single_data[x_index + 2]), fit_curve_list)
        # print([zip(*[(point_xy, np.poly1d(param)(point_xy)) for point_index, point_xy in enumerate(xy) if abs(np.poly1d(param)(point_xy) - single_data[x_index + 2][point_xy]) < 10]) for param in cluster_param_center])

        # compare_fit_curve((xy, single_data[x_index + 2]), [zip(*[(point_xy, fit_function(point_xy)) for point_index, point_xy in enumerate(xy) if abs(fit_function(point_xy) - single_data[x_index + 2][point_index]) < allowed_diff]) for fit_function in fit_function_list[::1000]])

# %%


