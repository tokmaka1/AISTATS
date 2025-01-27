import numpy as np
import torch
import torch.nn as nn
import tikzplotlib
import pickle
from scipy.special import comb
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gpytorch
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")


class MultiLayerRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(MultiLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define the first input branch RNN
        self.rnn1 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Define the second input branch RNN
        self.rnn2 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Merge layer
        self.merge_layer = nn.Linear(hidden_size * 2, hidden_size)
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x1, x2):
        # Forward pass for the first input branch
        out1, _ = self.rnn1(x1)
        # Forward pass for the second input branch
        out2, _ = self.rnn2(x2)
        # Concatenate the outputs of both branches
        out = torch.cat((out1, out2), dim=1)
        # Merge layer
        out = self.merge_layer(out)
        # Output layer
        out = self.fc(out)
        return out


def load_model(model_path, hidden_size, num_layers, num_classes):
    model = MultiLayerRNN(hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


def predict(model, input1, input2):
    model.eval()
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    with torch.no_grad():
        output = model(input1_tensor, input2_tensor)
    return output.item()


class GPRegressionModel(gpytorch.models.ExactGP):  # this model has to be build "new"
    def __init__(self, train_x, train_y, noise_std, n_devices=1, output_device=torch.device('cpu')):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(noise_std**2)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        self.kernel.lengthscale = 0.1
        # self.base_kernel.lengthscale.requires_grad = False; somehow does not work
        if output_device.type != 'cpu':
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                self.kernel, device_ids=range(n_devices), output_device=output_device)
        else:
            self.covar_module = self.kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ground_truth():
    def __init__(self, num_center_points, X_plot, RKHS_norm):
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        self.X_plot = X_plot
        self.RKHS_norm = RKHS_norm
        random_indices_center = torch.randint(high=self.X_plot.shape[0], size=(num_center_points,))
        self.X_center = self.X_plot[random_indices_center]
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        Kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        Kernel.lengthscale = 0.1
        RKHS_norm_squared = alpha.T @ Kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.f = fun(Kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)

    def conduct_experiment(self, x, noise_std):
        return torch.tensor(self.f(x) + np.random.normal(loc=0, scale=noise_std, size=1), dtype=x.dtype)


def save_data_for_RNN_training(K, var, noise_std, Y_sample, RKHS_norm_mean_function_list, vi_frac_list):  # We can also just do it for the cube from which we sample, and of course global. But let's see.
        alpha = torch.inverse(K+noise_std**2*torch.eye(K.shape[0])) @ Y_sample
        RKHS_norm_mean_function_list.append(torch.sqrt(alpha.reshape(1, -1) @ K @ alpha.reshape(-1, 1)).flatten()) # RKHS norm of the mean.
        variance_integral = sum(var)/2*(1/1000)
        vi_frac_list.append(1/variance_integral)
        return vi_frac_list, RKHS_norm_mean_function_list


def PAC_RKHS_norm_NeurIPS_numerical_investigation(X_plot, X_sample, Y_sample, N_hat, kernel, alpha_bar, gamma_PAC, kappa_PAC, m_PAC, RKHS_norm_mean_function_list, vi_frac_list, old_B):
    B = predict(RNN_model, RKHS_norm_mean_function_list, vi_frac_list)
    list_random_RKHS_norms = []
    for _ in range(m_PAC):
        x_interpol = X_sample
        y_interpol = Y_sample
        X_c = (min(X_plot) - max(X_plot))*torch.rand(N_hat, 1) + max((X_plot))
        X_c_tail = X_c[x_interpol.shape[0]:]
        X_c[:X_sample.shape[0]] = x_interpol
        alpha_tail = -2*alpha_bar*torch.rand(N_hat-x_interpol.shape[0], 1) + alpha_bar
        y_tail = kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
        y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
        alpha_head = torch.inverse(kernel(x_interpol, x_interpol).evaluate() + 1e-4*torch.eye(len(y_interpol))) @ y_head
        alpha = torch.cat((alpha_head, alpha_tail))
        random_RKHS_norm = torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha)
        list_random_RKHS_norms.append(random_RKHS_norm)
    numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
    numpy_list.sort()
    r_final = 0
    for r in range(m_PAC):  # you can compute this a priori
        summ = 0
        for i in range(r):
            summ += comb(m_PAC, i)*gamma_PAC**(i)*(1-gamma_PAC)**(m_PAC-i)
        if summ > kappa_PAC or B > numpy_list[-1-r]:
            break
        else:
            r_final = r
    B = max(B, numpy_list[-1-r_final])
    B = min(B, old_B)
    return B


def compute_PAC_RKHS_functions_NeurIPS(X_plot, X_sample, Y_sample, N_hat, kernel, alpha_bar, m_PAC):
    list_random_RKHS_functions = []
    for _ in range(m_PAC):
        x_interpol = X_sample
        y_interpol = Y_sample
        X_c = (min(X_plot) - max(X_plot))*torch.rand(N_hat, X_sample.shape[1]) + max((X_plot))
        X_c_tail = X_c[x_interpol.shape[0]:]
        X_c[:X_sample.shape[0]] = x_interpol
        alpha_tail = -2*alpha_bar*torch.rand(N_hat-x_interpol.shape[0], x_interpol.shape[1]) + alpha_bar
        y_tail = kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
        y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
        alpha_head = torch.inverse(kernel(x_interpol, x_interpol).evaluate() + 1e-4*torch.eye(len(y_interpol))) @ y_head
        alpha = torch.cat((alpha_head, alpha_tail))
        random_RKHS_function = kernel(X_plot, X_c).evaluate() @ alpha
        list_random_RKHS_functions.append(random_RKHS_function)
    return list_random_RKHS_functions



def compute_PAC_RKHS_norm_SysDO(X_plot, X_sample, Y_sample, N_hat, kernel, alpha_bar, q):
    list_random_RKHS_functions = []
    list_random_RKHS_norms = []
    for counter in range(q):
        x_interpol = X_sample
        y_interpol = Y_sample
        X_c = (min(X_plot) - max(X_plot))*torch.rand(N_hat, X_sample.shape[1]) + max((X_plot))
        X_c_tail = X_c[x_interpol.shape[0]:]
        X_c[:X_sample.shape[0]] = x_interpol
        alpha_tail = -2*alpha_bar*torch.rand(N_hat-x_interpol.shape[0], x_interpol.shape[1]) + alpha_bar
        y_tail = kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
        y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
        alpha_head = torch.inverse(kernel(x_interpol, x_interpol).evaluate()+torch.eye(len(y_interpol))*1e-5) @ y_head
        alpha = torch.cat((alpha_head, alpha_tail))
        if counter < 30:
            random_RKHS_function = kernel(X_plot, X_c).evaluate() @ alpha
            list_random_RKHS_functions.append(random_RKHS_function)
        random_RKHS_norm = torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha)
        list_random_RKHS_norms.append(random_RKHS_norm)
    numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
    numpy_list.sort()
    maximum_coefficients = torch.cat((alpha_head, alpha_bar*torch.ones(N_hat-len(alpha_head), 1)))
    maximum_possible_RKHS_norm = torch.sqrt(maximum_coefficients.T @ torch.ones(N_hat, N_hat) @ maximum_coefficients)
    minimum_possible_RKHS_norm = torch.sqrt(Y_sample.T@kernel(X_sample,X_sample) @ Y_sample) # kernel interpolation
    b_Hoeffding = maximum_possible_RKHS_norm.detach().numpy()
    a_Hoeffding = minimum_possible_RKHS_norm.detach().numpy()
    return numpy_list, list_random_RKHS_functions, a_Hoeffding, b_Hoeffding

def compare_PAC_RKHS_norm_SysDO_NeurIPS(X_plot, X_sample, Y_sample, N_hat, kernel, alpha_bar, m, kappa, gamma):
    list_random_RKHS_norms = []
    # list_RKHS_function = []
    # counter = 0
    for counter in range(m):
        x_interpol = X_sample
        y_interpol = Y_sample
        X_c = (min(X_plot) - max(X_plot))*torch.rand(N_hat, X_sample.shape[1]) + max((X_plot))
        X_c_tail = X_c[x_interpol.shape[0]:]  # the end coefficients
        X_c[:X_sample.shape[0]] = x_interpol
        alpha_tail = -2*alpha_bar*torch.rand(N_hat-x_interpol.shape[0], x_interpol.shape[1]) + alpha_bar
        y_tail = kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
        y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
        alpha_head = torch.inverse(kernel(x_interpol, x_interpol).evaluate()+torch.eye(len(y_interpol))*1e-3) @ y_head
        alpha = torch.cat((alpha_head, alpha_tail))
        random_RKHS_norm = torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha)
        list_random_RKHS_norms.append(random_RKHS_norm)
        # if counter < 50:
        #     random_RKHS_function = kernel(X_plot, X_c).evaluate() @ alpha
        #     list_RKHS_function.append(random_RKHS_function)

    numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
    numpy_list.sort()  # sorted RKHS norms
    maximum_coefficients = torch.cat((alpha_head, alpha_bar*torch.ones(N_hat-len(alpha_head), 1)))
    maximum_possible_RKHS_norm = torch.sqrt(maximum_coefficients.T @ torch.ones(N_hat, N_hat) @ maximum_coefficients)
    minimum_possible_RKHS_norm = torch.sqrt(Y_sample.T@kernel(X_sample,X_sample) @ Y_sample) # kernel interpolation
    b_Hoeffding = maximum_possible_RKHS_norm.detach().numpy()
    a_Hoeffding = minimum_possible_RKHS_norm.detach().numpy()
    B_SysDO = np.mean(numpy_list) + np.sqrt(np.log(1/gamma)*(b_Hoeffding-a_Hoeffding)**2/(2*m))
    for r in range(m_PAC):  # you can compute this a priori
        summ = 0
        for i in range(r):
            summ += comb(m_PAC, i)*gamma_PAC**(i)*(1-gamma_PAC)**(m_PAC-i)
        if summ > kappa:
            break
    r_final = r - 1 if r > 0 else r
    B_NeurIPS = numpy_list[-1-r_final]
    return B_SysDO, B_NeurIPS


def scenario_approach_time(X_plot, X_sample, inverse_matrix, Y_sample, N_hat, kernel, alpha_bar, m, kappa_PAC, gamma_PAC):
    current_time = time.time()
    list_random_RKHS_norms = []
    # list_RKHS_function = []
    # counter = 0
    for _ in tqdm(range(m)):
        x_interpol = X_sample
        y_interpol = Y_sample
        X_c = (min(X_plot) - max(X_plot))*torch.rand(N_hat, X_sample.shape[1]) + max((X_plot))
        X_c_tail = X_c[x_interpol.shape[0]:]  # the end coefficients
        X_c[:X_sample.shape[0]] = x_interpol
        alpha_tail = -2*alpha_bar*torch.rand(N_hat-x_interpol.shape[0], x_interpol.shape[1]) + alpha_bar
        y_tail = kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
        y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
        alpha_head = inverse_matrix @ y_head
        alpha = torch.cat((alpha_head, alpha_tail))
        random_RKHS_norm = torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha)
        list_random_RKHS_norms.append(random_RKHS_norm)

    numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
    numpy_list.sort()  # sorted RKHS norms
    for r in range(m_PAC):  # you can compute this a priori
        summ = 0
        for i in range(r):
            summ += comb(m_PAC, i)*gamma_PAC**(i)*(1-gamma_PAC)**(m_PAC-i)
        if summ > kappa_PAC:
            break
    r_final = r - 1 if r > 0 else r
    time_required = time.time() - current_time  # total time needed
    return time_required


if __name__ == '__main__':
    plot_local_RKHS_norm = False
    plot_numerical_investigation = False
    numerical_RKHS_investigation = False
    compare_PACSBO_NeurIPS = False
    scenario_approach_numerical_investigation = False
    scenario_approach_time_investigation = True

    if plot_local_RKHS_norm:
        # Create artificial data to approximate
        x_values = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.6, 0.7, 0.8, 0.9, 1])
        y_values = torch.tensor([0, 0.05, -0.05, -0.02, -0.09, 0.1, 0.12, 0.13, 0.32, 0.4, 0.5, 0.8, 0.9, 1, 0.3, 0.1, 0.05, 0.11, 0.1, 0.11, 0.10])
        X_plot = torch.linspace(0, 1, 999).reshape(-1, 1)
        # Kernel interpolation for ground truth
        kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        kernel.lengthscale = 0.1
        cov_matrix_full = kernel(x_values, x_values).evaluate()
        cov_vector_full = kernel(X_plot, x_values).evaluate()
        hX_full = (y_values.reshape(1, -1) @ torch.inverse(cov_matrix_full) @ cov_vector_full.T).flatten()
        RKHS_norm = torch.sqrt(y_values.T @ torch.inverse(cov_matrix_full) @ y_values)
        # Divide the ground truth into 3 different parts
        X_plot_1 = torch.linspace(0, 1/3, 333).reshape(-1, 1)
        X_plot_2 = torch.linspace(1/3, 2/3, 333).reshape(-1, 1)
        X_plot_3 = torch.linspace(2/3, 1, 333).reshape(-1, 1)

        fX_1 = hX_full[:333]
        fX_2 = hX_full[333:666]
        fX_3 = hX_full[666:]

        cov_matrix_1 = kernel(X_plot_1, X_plot_1).evaluate()
        cov_vector_1 = kernel(X_plot_1, X_plot_1).evaluate()
        hX_1 = (fX_1.reshape(1, -1) @ torch.inverse(cov_matrix_1 + torch.eye(333)*1e-4) @ cov_vector_1.T).flatten()
        RKHS_norm_1 = torch.sqrt(fX_1.T @ torch.inverse(cov_matrix_1 + torch.eye(333)*1e-4) @ fX_1)


        cov_matrix_2 = kernel(X_plot_2, X_plot_2).evaluate()
        cov_vector_2 = kernel(X_plot_2, X_plot_2).evaluate()
        hX_2 = (fX_2.reshape(1, -1) @ torch.inverse(cov_matrix_2 + torch.eye(333)*1e-4) @ cov_vector_2.T).flatten()
        RKHS_norm_2 = torch.sqrt(fX_2.T @ torch.inverse(cov_matrix_2 + torch.eye(333)*1e-4) @ fX_2)


        cov_matrix_3 = kernel(X_plot_3, X_plot_3).evaluate()
        cov_vector_3 = kernel(X_plot_3, X_plot_3).evaluate()
        hX_3 = (fX_3.reshape(1, -1) @ torch.inverse(cov_matrix_3 + torch.eye(333)*1e-4) @ cov_vector_3.T).flatten()
        RKHS_norm_3 = torch.sqrt(fX_3.T @ torch.inverse(cov_matrix_3 + torch.eye(333)*1e-4) @ fX_3)

        print(RKHS_norm, RKHS_norm_1, RKHS_norm_2, RKHS_norm_3)

        plt.figure()
        plt.plot(X_plot, hX_full.detach().numpy())
        tikzplotlib.save('local_RKHS_norms.tex')


    elif numerical_RKHS_investigation:
        # NeurIPS, numerical investigation
        model_path = "rnn_model.pt"
        hidden_size = 20
        num_layers = 2
        num_classes = 1
        RNN_model = load_model(model_path, hidden_size, num_layers, num_classes)
        m_PAC = 1000
        kappa_PAC = 0.01
        alpha_bar = 1
        gamma_PAC = 0.1
        N_hat = 500
        noise_std = 0.01
        # list_all_results = []
        dict_all_results = {}
        for i in tqdm(range(100)):  # 250 random RKHS functions
            list_results = []
            X_plot = torch.linspace(0, 1, 1000).reshape(-1, 1)
            RKHS_norm = np.random.uniform(low=1, high=10)
            gt = ground_truth(num_center_points=np.random.choice(range(100, 1000)), X_plot=X_plot, RKHS_norm=RKHS_norm)
            # kernel = gt.kernel
            RKHS_norm_mean_function_list = []
            vi_frac_list = []
            X_sample = None
            B = None
            for n_samples in range(1, 30):
                x_new = X_plot[torch.randperm(len(X_plot))[0]].reshape(-1, 1)
                # random_indices = torch.cat((random_indices_old, random_index)) if random_indices_old is not None else random_index
                if X_sample is None:
                    X_sample = torch.tensor(x_new)
                else:
                    X_sample = torch.cat((x_new, X_sample))
                Y_sample = gt.conduct_experiment(x=X_sample, noise_std=noise_std)
                # (len(Y_sample))
                model = GPRegressionModel(train_x=X_sample, train_y=Y_sample, noise_std=noise_std)
                model.eval()
                f_preds = model(X_plot)
                mean = f_preds.mean
                var = f_preds.variance
                K = model(X_sample).covariance_matrix
                RKHS_norm_mean_function_list, vi_frac_list = save_data_for_RNN_training(K, var, noise_std, Y_sample, RKHS_norm_mean_function_list, vi_frac_list)
                if B is None:
                    old_B = np.infty
                else:
                    old_B = B
                B = PAC_RKHS_norm_NeurIPS_numerical_investigation(X_plot, X_sample, Y_sample, N_hat, model.kernel, alpha_bar, gamma_PAC, kappa_PAC, m_PAC, RKHS_norm_mean_function_list, vi_frac_list, old_B)
                list_results.append(B/RKHS_norm)
            dict_all_results[RKHS_norm] = list_results
        with open('dict_all_results_100_30_in_1_10.pickle', 'wb') as handle:
            pickle.dump(dict_all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif plot_numerical_investigation:
        with open('Experiments/numerical_investigation/dict_all_results_100_30_in_1_10.pickle', 'rb') as handle:
            dict_all_results_1 = pickle.load(handle)
        with open('Experiments/numerical_investigation/dict_all_results_100_30_in_1_10_run2.pickle', 'rb') as handle:
            dict_all_results_2 = pickle.load(handle)
        list_results_1 = list(dict_all_results_1.values())
        list_results_2 = list(dict_all_results_2.values())
        list_results = list_results_1 + list_results_2
        counter = 0
        for value in list_results:
            if min(value) < 1:
                counter += 1
        print(f'{counter} out of {len(list_results)} have under-estimated the RKHS norm.')
        list_mean = []
        list_std = []
        for i in range(len(list_results[0])):  # all of them have the same length anyways. We want to know the value of the ith over-estimation for all functions
            all_values = []
            for j in range(len(list_results)):  # now iteration through all functions  
                all_values.append(list_results[j][i])
            list_mean.append(np.mean(all_values))
            list_std.append(np.std(all_values))
        upper_bound = [m + s for m, s in zip(list_mean, list_std)]
        lower_bound = [m - s for m, s in zip(list_mean, list_std)]
        plt.figure()
        plt.plot(range(1, len(list_mean)+1), list_mean, color='black')
        plt.fill_between(range(1, len(list_mean)+1), lower_bound, upper_bound, color='lightblue', alpha=0.5)
        plt.plot(range(1, len(list_mean)+1), [1]*len(list_mean), 'r')
        plt.xlabel('Iteration $t$')
        plt.ylabel('Ratio')
        plt.savefig('numerical_investigation.png')
        tikzplotlib.clean_figure()
        tikzplotlib.save('numerical_investigation.tex')

    elif compare_PACSBO_NeurIPS:

        delta = 0.1
        m_PAC_list = [1000, 2000, 3000, 4000, 5000]
        dict_comparison_SysDO_NeurIPS = {}
        for m_PAC in tqdm(m_PAC_list):
            dict_comparison_SysDO_NeurIPS[m_PAC] = {}
            X_plot = torch.linspace(0, 1, 1000).reshape(-1, 1)
            gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=5)
            # kernel = gt.kernel
            kernel = gpytorch.kernels.MaternKernel(nu=1.5)
            kernel.lengthscale = 0.1
            RKHS_norm_mean_function_list = []
            vi_frac_list = []
            noise_std = 1e-2
            N_hat = 500
            alpha_bar = 1
            gamma_PAC = 0.1
            kappa_PAC = 0.0001

            random_indices_5 = torch.randperm(len(X_plot))[:5]
            X_sample_5 = X_plot[random_indices_5]
            Y_sample_5 = torch.tensor(gt.f(X_sample_5), dtype=torch.float32)
            B_SysDO_5, B_NeurIPS_5 = compare_PAC_RKHS_norm_SysDO_NeurIPS(X_plot, X_sample_5, Y_sample_5, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

            random_indices_15 = torch.randperm(len(X_plot))[:15]
            random_indices_20 = torch.cat((random_indices_5, random_indices_15))
            X_sample_20 = X_plot[random_indices_20]
            Y_sample_20 = torch.tensor(gt.f(X_sample_20), dtype=torch.float32)
            B_SysDO_20, B_NeurIPS_20 = compare_PAC_RKHS_norm_SysDO_NeurIPS(X_plot, X_sample_20, Y_sample_20, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

            random_indices_30 = torch.randperm(len(X_plot))[:30]
            random_indices_50 = torch.cat((random_indices_20, random_indices_30))
            X_sample_50 = X_plot[random_indices_50]
            Y_sample_50 = torch.tensor(gt.f(X_sample_50), dtype=torch.float32)
            B_SysDO_50, B_NeurIPS_50 = compare_PAC_RKHS_norm_SysDO_NeurIPS(X_plot, X_sample_50, Y_sample_50, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)
            dict_comparison_SysDO_NeurIPS[m_PAC]['SysDO'] = [B_SysDO_5, B_SysDO_20, B_SysDO_50]
            dict_comparison_SysDO_NeurIPS[m_PAC]['NeurIPS'] = [B_NeurIPS_5, B_NeurIPS_20, B_NeurIPS_50]
        print('Done!')
    elif scenario_approach_numerical_investigation:
        # fix kappa and gamma, see how m and r correlate
        kappa = 0.001
        gamma = 0.01
        list_r = []
        # m_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 
        #           1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
        #           1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
        m_list = [x*500 for x in range(1, 15)]
        for m in tqdm(m_list):
            for r in range(m):  # try different r
                summ = 0  # now r is fixed
                for i in range(r+1):  # compute the sum over r
                    summ += comb(m, i)*gamma**(i)*(1-gamma)**(m-i)
                if summ > kappa:
                    break
                # else:
                #     r_final = r
            list_r.append(r-1 if r > 0 else 0)
        plt.figure()
        plt.plot(m_list, list_r)
        plt.xlabel('m')
        plt.ylabel('r')
        plt.title('kappa=0.001, gamma=0.01')
        plt.savefig('gamma_kappa_fixed.png')
        tikzplotlib.save('gamma_kappa_fixed.tex')
        # fix kappa and m, see how gamma and r correlate
        m = 2500
        list_r = []
        list_gamma = [0.001*x for x in range(1, 100)]
        for gamma in list_gamma:
            for r in range(m):  # try different r
                summ = 0  # now r is fixed
                for i in range(r+1):  # compute the sum over r
                    summ += comb(m, i)*gamma**(i)*(1-gamma)**(m-i)
                if summ > kappa:
                    break
                # else:
                #     r_final = r
            list_r.append(r-1 if r > 0 else 0)
        plt.figure()
        plt.plot(list_gamma, list_r)
        plt.xlabel('gamma')
        plt.ylabel('r')
        plt.title('m=2500, kappa=0.001')
        plt.savefig('m_kappa_fixed.png')
        tikzplotlib.save('m_kappa_fixed.tex')
        m_list = [x*500 for x in range(1, 15)]
        r = 0
        kappa_list = []
        gamma = 0.01
        for m in m_list:
            kappa = (1-gamma)**m
            kappa_list.append(kappa)
        
        plt.plot(m_list, kappa_list)
        plt.yscale('log')
        plt.xlabel('m')
        plt.ylabel('kappa (log)')
        plt.savefig('gamma_r_fixed.png')
        tikzplotlib.save('gamma_r_fixed.tex')
        print('hallo')
    elif scenario_approach_time_investigation:
        get_data = False
        if not get_data:
            with open('dict_scenario_time.pickle', 'rb') as handle:
                dict_scenario_time_1 = pickle.load(handle)
            with open('dict_scenario_time_2.pickle', 'rb') as handle:
                dict_scenario_time_2 = pickle.load(handle)
            with open('dict_scenario_time_3.pickle', 'rb') as handle:
                dict_scenario_time_3 = pickle.load(handle)
            with open('dict_scenario_time_4.pickle', 'rb') as handle:
                dict_scenario_time_4 = pickle.load(handle)
            with open('dict_scenario_time_5.pickle', 'rb') as handle:
                dict_scenario_time_5 = pickle.load(handle)
            with open('dict_scenario_time_6.pickle', 'rb') as handle:
                dict_scenario_time_6 = pickle.load(handle)
            with open('dict_scenario_time_7.pickle', 'rb') as handle:
                dict_scenario_time_7 = pickle.load(handle)
            with open('dict_scenario_time_8.pickle', 'rb') as handle:
                dict_scenario_time_8 = pickle.load(handle)
            with open('dict_scenario_time_9.pickle', 'rb') as handle:
                dict_scenario_time_9 = pickle.load(handle)
            with open('dict_scenario_time_10.pickle', 'rb') as handle:
                dict_scenario_time_10 = pickle.load(handle)
            scenario_time_500 = np.array([dict_scenario_time_1[500], dict_scenario_time_2[500], dict_scenario_time_3[500], dict_scenario_time_4[500], dict_scenario_time_5[500], dict_scenario_time_6[500], dict_scenario_time_7[500], dict_scenario_time_8[500], dict_scenario_time_9[500], dict_scenario_time_10[500]])
            scenario_time_1000 = np.array([dict_scenario_time_1[1000], dict_scenario_time_2[1000], dict_scenario_time_3[1000], dict_scenario_time_4[1000], dict_scenario_time_5[1000], dict_scenario_time_6[1000], dict_scenario_time_7[1000], dict_scenario_time_8[1000], dict_scenario_time_9[1000], dict_scenario_time_10[1000]])
            scenario_time_2000 = np.array([dict_scenario_time_1[2000], dict_scenario_time_2[2000], dict_scenario_time_3[2000], dict_scenario_time_4[2000], dict_scenario_time_5[2000], dict_scenario_time_6[2000], dict_scenario_time_7[2000], dict_scenario_time_8[2000], dict_scenario_time_9[2000], dict_scenario_time_10[2000]])
            scenario_time_3000 = np.array([dict_scenario_time_1[3000], dict_scenario_time_2[3000], dict_scenario_time_3[3000], dict_scenario_time_4[3000], dict_scenario_time_5[3000], dict_scenario_time_6[3000], dict_scenario_time_7[3000], dict_scenario_time_8[3000], dict_scenario_time_9[3000], dict_scenario_time_10[3000]])
            mean_500, std_500 = np.mean(scenario_time_500, axis=0), np.std(scenario_time_500, axis=0)
            mean_1000, std_1000 = np.mean(scenario_time_1000, axis=0), np.std(scenario_time_1000, axis=0)
            mean_2000, std_2000 = np.mean(scenario_time_2000, axis=0), np.std(scenario_time_2000, axis=0)
            mean_3000, std_3000 = np.mean(scenario_time_3000, axis=0), np.std(scenario_time_3000, axis=0)
            iterations = [5, 20, 50, 100, 200, 300]

        if get_data:
            delta = 0.1
            m_PAC_list = [500, 1000, 2000, 3000]
            dict_scenario_time = {}
            for m_PAC in m_PAC_list:
                X_plot = torch.linspace(0, 1, 1000).reshape(-1, 1)
                gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=5)
                # kernel = gt.kernel
                kernel = gpytorch.kernels.MaternKernel(nu=1.5)
                kernel.lengthscale = 0.1
                RKHS_norm_mean_function_list = []
                vi_frac_list = []
                noise_std = 1e-2
                N_hat = 500
                alpha_bar = 1
                gamma_PAC = 0.1
                kappa_PAC = 0.01

                random_indices_5 = torch.randperm(len(X_plot))[:5]
                X_sample_5 = X_plot[random_indices_5]
                inverse_matrix = torch.inverse(kernel(X_sample_5, X_sample_5).evaluate() + 1e-4*torch.eye(5))
                Y_sample_5 = torch.tensor(gt.f(X_sample_5), dtype=torch.float32)
                t_5 = scenario_approach_time(X_plot, X_sample_5, inverse_matrix, Y_sample_5, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

                random_indices_15 = torch.randperm(len(X_plot))[:15]
                random_indices_20 = torch.cat((random_indices_5, random_indices_15))
                X_sample_20 = X_plot[random_indices_20]
                inverse_matrix = torch.inverse(kernel(X_sample_20, X_sample_20).evaluate() + 1e-4*torch.eye(20))
                Y_sample_20 = torch.tensor(gt.f(X_sample_20), dtype=torch.float32)
                t_20 = scenario_approach_time(X_plot, X_sample_20, inverse_matrix, Y_sample_20, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

                random_indices_30 = torch.randperm(len(X_plot))[:30]
                random_indices_50 = torch.cat((random_indices_20, random_indices_30))
                X_sample_50 = X_plot[random_indices_50]
                inverse_matrix = torch.inverse(kernel(X_sample_50, X_sample_50).evaluate() + 1e-4*torch.eye(50))
                Y_sample_50 = torch.tensor(gt.f(X_sample_50), dtype=torch.float32)
                t_50 = scenario_approach_time(X_plot, X_sample_50, inverse_matrix, Y_sample_50, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

                random_indices_50_ = torch.randperm(len(X_plot))[:50]
                random_indices_100 = torch.cat((random_indices_50, random_indices_50_))
                X_sample_100 = X_plot[random_indices_100]
                inverse_matrix = torch.inverse(kernel(X_sample_100, X_sample_100).evaluate() + 1e-4*torch.eye(100))
                Y_sample_100 = torch.tensor(gt.f(X_sample_100), dtype=torch.float32)
                t_100 = scenario_approach_time(X_plot, X_sample_100, inverse_matrix, Y_sample_100, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

                random_indices_100_ = torch.randperm(len(X_plot))[:100]
                random_indices_200 = torch.cat((random_indices_100, random_indices_100_))
                X_sample_200 = X_plot[random_indices_200]
                inverse_matrix = torch.inverse(kernel(X_sample_200, X_sample_200).evaluate() + 1e-4*torch.eye(200))
                Y_sample_200 = torch.tensor(gt.f(X_sample_200), dtype=torch.float32)
                t_200 = scenario_approach_time(X_plot, X_sample_200, inverse_matrix, Y_sample_200, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)

                random_indices_100__ = torch.randperm(len(X_plot))[:100]
                random_indices_300 = torch.cat((random_indices_200, random_indices_100__))
                X_sample_300 = X_plot[random_indices_300]
                inverse_matrix = torch.inverse(kernel(X_sample_300, X_sample_300).evaluate() + 1e-4*torch.eye(300))
                Y_sample_300 = torch.tensor(gt.f(X_sample_300), dtype=torch.float32)
                t_300 = scenario_approach_time(X_plot, X_sample_300, inverse_matrix, Y_sample_300, N_hat, kernel, alpha_bar, m_PAC, kappa_PAC, gamma_PAC)


                dict_scenario_time[m_PAC] = [t_5, t_20, t_50, t_100, t_200, t_300]

            print('Done')
            with open('dict_scenario_time_6.pickle', 'wb') as handle:
                pickle.dump(dict_scenario_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
            raise Exception('Done')
    # NeurIPS
    X_plot = torch.linspace(0, 1, 1000).reshape(-1, 1)
    random_indices_1 = torch.randperm(len(X_plot))[:1]
    gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=5)
    kernel = gt.kernel
    noise_std = 0.01

    X_sample_1 = X_plot[random_indices_1]
    Y_sample_1 = gt.conduct_experiment(X_sample_1, noise_std)
    list_random_RKHS_functions_1 = compute_PAC_RKHS_functions_NeurIPS(X_plot, X_sample_1, Y_sample_1, N_hat=500, kernel=kernel, alpha_bar=1, m_PAC=25)

    random_indices_9 = torch.randperm(len(X_plot))[:9]
    random_indices_10 = torch.cat((random_indices_1, random_indices_9))
    X_sample_10 = X_plot[random_indices_10]
    Y_sample_10 = gt.conduct_experiment(X_sample_10, noise_std)
    list_random_RKHS_functions_10 = compute_PAC_RKHS_functions_NeurIPS(X_plot, X_sample_10, Y_sample_10, N_hat=500, kernel=kernel, alpha_bar=1, m_PAC=25)

    random_indices_20 = torch.randperm(len(X_plot))[:20]
    random_indices_30 = torch.cat((random_indices_10, random_indices_20))
    X_sample_30 = X_plot[random_indices_30]
    Y_sample_30 = gt.conduct_experiment(X_sample_30, noise_std)
    list_random_RKHS_functions_30 = compute_PAC_RKHS_functions_NeurIPS(X_plot, X_sample_30, Y_sample_30, N_hat=500, kernel=kernel, alpha_bar=1, m_PAC=25)

    plt.figure()
    for random_RKHS_function in list_random_RKHS_functions_1:
        plt.plot(X_plot, random_RKHS_function.detach().numpy(), color='magenta', alpha=0.10)
    tikzplotlib.clean_figure()
    plt.scatter(X_sample_1, Y_sample_1, color='black')
    tikzplotlib.save('random_functions_1.tex')
    plt.savefig('random_functions_1.png')

    plt.figure()
    for random_RKHS_function in list_random_RKHS_functions_10:
        plt.plot(X_plot, random_RKHS_function.detach().numpy(), color='magenta', alpha=0.10)
    plt.plot(X_plot, gt.f(X_plot), 'blue')
    tikzplotlib.clean_figure()
    plt.scatter(X_sample_10, Y_sample_10, color='black')
    tikzplotlib.save('random_functions_10.tex')
    plt.savefig('random_functions_10.png')

    plt.figure()
    for random_RKHS_function in list_random_RKHS_functions_30:
        plt.plot(X_plot, random_RKHS_function.detach().numpy(), color='magenta', alpha=0.10)
    plt.plot(X_plot, gt.f(X_plot), 'blue')
    tikzplotlib.clean_figure()
    plt.scatter(X_sample_30, Y_sample_30, color='black')
    tikz_code = tikzplotlib.get_tikz_code(float_format=".6g")
    tikzplotlib.save('random_functions_30.tex')
    plt.savefig('random_functions_30.png')

    # SysDO
    delta = 0.1
    q = 5000
    X_plot = torch.linspace(0, 1, 1000).reshape(-1, 1)
    gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=1)
    kernel = gt.kernel

    random_indices_5 = torch.randperm(len(X_plot))[:5]
    X_sample_5 = X_plot[random_indices_5]
    Y_sample_5 = torch.tensor(gt.f(X_sample_5), dtype=torch.float32)
    numpy_list_5, list_random_RKHS_functions_5, a_Hoeffding_5, b_Hoeffding_5 = compute_PAC_RKHS_norm_SysDO(X_plot, X_sample_5, Y_sample_5, N_hat=100, kernel=kernel, alpha_bar=1, q=q)
    PAC_RKHS_norm_5 = np.mean(numpy_list_5) + np.sqrt(np.log(1/delta)*(b_Hoeffding_5-a_Hoeffding_5)**2/(2*q))
    print(f'For 5 samples, the PAC RKHS norm is: {PAC_RKHS_norm_5}')

    random_indices_15 = torch.randperm(len(X_plot))[:15]
    random_indices_20 = torch.cat((random_indices_5, random_indices_15))
    X_sample_20 = X_plot[random_indices_20]
    Y_sample_20 = torch.tensor(gt.f(X_sample_20), dtype=torch.float32)
    numpy_list_20, list_random_RKHS_functions_20, a_Hoeffding_20, b_Hoeffding_20 = compute_PAC_RKHS_norm_SysDO(X_plot, X_sample_20, Y_sample_20, N_hat=100, kernel=kernel, alpha_bar=1, q=q)
    PAC_RKHS_norm_20 = np.mean(numpy_list_20) + np.sqrt(np.log(1/delta)*(b_Hoeffding_20-a_Hoeffding_20)**2/(2*q))
    print(f'For 20 samples, the PAC RKHS norm is: {PAC_RKHS_norm_20}')

    random_indices_30 = torch.randperm(len(X_plot))[:30]
    random_indices_50 = torch.cat((random_indices_20, random_indices_30))
    X_sample_50 = X_plot[random_indices_50]
    Y_sample_50 = torch.tensor(gt.f(X_sample_50), dtype=torch.float32)
    numpy_list_50, list_random_RKHS_functions_50, a_Hoeffding_50, b_Hoeffding_50 = compute_PAC_RKHS_norm_SysDO(X_plot, X_sample_50, Y_sample_50, N_hat=100, kernel=kernel, alpha_bar=1, q=q)
    PAC_RKHS_norm_50 = np.mean(numpy_list_50) + np.sqrt(np.log(1/delta)*(b_Hoeffding_50-a_Hoeffding_50)**2/(2*q))
    print(f'For 50 samples, the PAC RKHS norm is: {PAC_RKHS_norm_50}')

    plt.figure()
    for random_RKHS_function in list_random_RKHS_functions_5:
        plt.plot(X_plot, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue')
    plt.plot(X_plot, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue', label='Random RKHS functions')
    plt.scatter(X_sample_5, Y_sample_5, color='black', s=100, label='Samples')
    plt.plot(X_plot, gt.fX, color='red', label='Ground truth')
    tikzplotlib.clean_figure()
    tikzplotlib.save('PAC_functions_5.tex')

    plt.figure()
    for random_RKHS_function in list_random_RKHS_functions_20:
        plt.plot(X_plot, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue')
    plt.plot(X_plot, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue', label='Random RKHS functions')
    plt.scatter(X_sample_20, Y_sample_20, color='black', s=100, label='Samples')
    plt.plot(X_plot, gt.fX, color='red', label='Ground truth')
    tikzplotlib.clean_figure()
    tikzplotlib.save('PAC_functions_20.tex')

    plt.figure()
    for random_RKHS_function in list_random_RKHS_functions_50:
        plt.plot(X_plot, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue')
    plt.plot(X_plot, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue', label='Random RKHS functions')
    plt.scatter(X_sample_50, Y_sample_50, color='black', s=100, label='Samples')
    plt.plot(X_plot, gt.fX, color='red', label='Ground truth')
    tikzplotlib.clean_figure()
    tikzplotlib.save('PAC_functions_50.tex')

    rounded_values_5 = np.round(numpy_list_5, decimals=1)
    min_val_5 = min(numpy_list_5)
    max_val_5 = max(numpy_list_5)
    bin_range_5 = max_val_5 * 1.1 - min_val_5 * 0.9

    rounded_values_20 = np.round(numpy_list_20, decimals=1)
    min_val_20 = min(numpy_list_20)
    max_val_20 = max(numpy_list_20)
    bin_range_20 = max_val_20 * 1.1 - min_val_20 * 0.9

    rounded_values_50 = np.round(numpy_list_50, decimals=2)
    min_val_50 = min(numpy_list_50)
    max_val_50 = max(numpy_list_50)
    bin_range_50 = max_val_50 * 1.1 - min_val_50 * 0.9

    plt.figure()
    plt.hist(rounded_values_5, bins=np.arange(min_val_5 * 0.9, max_val_5 * 1.1, bin_range_5 / 10), edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random RKHS Norms, 5 samples')
    plt.grid(False)
    plt.show()
    tikzplotlib.save('PAC_norms_5.tex')


    plt.figure()
    plt.hist(rounded_values_20, bins=np.arange(min_val_20 * 0.9, max_val_20 * 1.1, bin_range_20 / 10), edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random RKHS Norms, 20 samples')
    plt.grid(False)
    plt.show()
    tikzplotlib.save('PAC_norms_20.tex')

    plt.figure()
    plt.hist(rounded_values_50, bins=np.arange(min_val_50 * 0.9, max_val_50 * 1.1, bin_range_50 / 10), edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random RKHS Norms, 50 samples')
    plt.grid(False)
    plt.show()
    tikzplotlib.save('PAC_norms_50.tex')
