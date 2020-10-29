import tensorflow as tf
from halo import Halo
from keras.utils.layer_utils import count_params

import numpy as np
import pickle
import pandas as pd
from scipy.optimize import differential_evolution
from colorama import Fore, Style

from makeTrainNet import DeepNetBonanza
from time import strftime, gmtime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DiffEvolPickArch:
    """
        Class to pick out optimum architeture via differential evolution
    """

    def __init__(self, train_dict, param_pos_vals, max_iter,
                 name='Enkidu', de_loss_dict=None,
                 mb_size=128, nb_epochs=200, de_max_iter=3, out_layer_act_fct='linear',
                 avg_nb=1,
                 verbose=1):
        """

        """
        self.res_loss = tf.keras.losses.MeanAbsoluteError()
        self.train_dict = train_dict
        self.max_iter = max_iter
        self.mb_size = mb_size
        self.epochs = nb_epochs
        self.max_iter = de_max_iter
        self.param_pos_vals = param_pos_vals
        self.out_layer_act_fct = out_layer_act_fct
        self.verbose = verbose
        self._make_param_map()
        self.avg_nb = avg_nb
        self.name = name

        if not de_loss_dict:
            #
            # min_model = DeepNetBonanza(self.param_pos_vals[0][0], self.param_pos_vals[1][0], self.train_dict,
            #                            verbose=0)
            # min_nb_p = count_params(min_model.model.trainable_weights)

            # max_model = DeepNetBonanza(self.param_pos_vals[0][-1], self.param_pos_vals[1][-1], self.train_dict,
            #                            verbose=0)
            # max_nb_p = count_params(max_model.model.trainable_weights)

            self.de_loss = {'Res_Loss': tf.keras.losses.MeanAbsoluteError(),
                            'Param_Loss': lambda x: np.log(x)
                            # 'Param_Loss': lambda x: (x - min_nb_p) / (max_nb_p - min_nb_p)
                            }
            # tf.keras.backend.clear_session()
            # del min_model, max_model

        elif isinstance(de_loss_dict, dict):
            self.de_loss = de_loss_dict

        self.session_ID = 'ID_' + strftime("%d-%m-%Y-%H_%M_%S", gmtime())
        log_loss = {'': []}
        dat_frame = pd.DataFrame(columns=['nb_nerons', 'nb_layers', 'act_fct', 'de_loss_avg', 'param_loss'])
        log_dir = 'DE_Loss_Logs/'
        os.makedirs(log_dir, exist_ok=True)
        dat_frame.to_csv(f"{log_dir}{self.name}_{self.session_ID}.csv", mode='w')

    def _conv_net_vec(self, arch_param_vec):
        """
            Converts the numerical xVecArch into the discrete version.

            :param arch_param_vec:
            :return:
        """
        net_arch_vec = []

        for comp_nb, arch_comp in enumerate(arch_param_vec):
            for bound_idx in range(len(self.param_map[comp_nb]) - 1):
                bound_l = self.param_map[comp_nb][bound_idx]
                bound_r = self.param_map[comp_nb][bound_idx + 1]
                # print(f'Fitting {arch_comp} against [{bound_l},{bound_r}]')
                if bound_l <= arch_comp < bound_r:
                    # print(f'Choosing {self.param_pos_vals[comp_nb][bound_idx]} out of {self.param_pos_vals[comp_nb]}')
                    net_arch_vec.append(self.param_pos_vals[comp_nb][bound_idx])
                    break

            else:
                if arch_comp >= 1:
                    # corresponds to the component being equal to the rhs end point == bound_r
                    net_arch_vec.append(self.param_pos_vals[comp_nb][-1])

        return net_arch_vec

    def _aux_min_func(self, arch_param_vec):
        """
            Auxiliary minimisation function for the architecture picker.
            :return:
        """
        # print('\n---------------------:',xVecArch, '\n---------------------:')
        arch_vec = self._conv_net_vec(arch_param_vec)
        nb_nerons, nb_layers, act_fct = arch_vec[0], arch_vec[1], arch_vec[2]
        arch_str = f'N{nb_nerons}_L{nb_layers}_{act_fct}'

        res_loss_sum = 0
        param_loss = np.inf
        # de_loss_list = []

        fill_str = '-' * 50
        print('\n' + fill_str + f' Averaging over training for {arch_str} ' + fill_str)
        # spinner = Halo(text=f'Starting Test 0 for {arch_str}', spinner='dots')
        # spinner.start()
        tf.keras.backend.clear_session()
        for try_nb in range(self.avg_nb):
            stat_str = f"Test: {try_nb + 1}/{self.avg_nb} || Trying {arch_str} with" \
                       f" a MSize:{self.mb_size} for {self.epochs} Epochs. "

            # ---------- Initialise and train the new network ----------
            net_inst = DeepNetBonanza(nb_nerons, nb_layers, train_dict=self.train_dict,
                                      drop_rate=0.2, verbose=self.verbose, name=self.name
                                      )
            net_inst.train_net(nb_epochs=self.epochs, mb_size=self.mb_size)
            # ---------- Get and return the DE evolution loss ----------
            res_loss_part, param_loss, loss_dict = self.get_de_loss(net_inst.model)
            del net_inst

            # ---------- Update the statistics string ----------
            stat_str += ' ---> '
            for loss_key in loss_dict:
                stat_str += f'{loss_key} : {loss_dict[loss_key]:.4f}  |  '
            stat_str += f'Comb DE Loss: {res_loss_part * param_loss:.4f}'
            # spinner.text = stat_str
            # de_loss_list.append(de_loss_part)
            res_loss_sum += res_loss_part

            print(stat_str)
        de_loss_avg = res_loss_sum / self.avg_nb
        de_loss_avg = de_loss_avg * param_loss
        # spinner.stop_and_persist(symbol='⠿', text=stat_str)

        # de_loss_arr = np.array(de_loss_list)
        #
        # de_hist, de_bins = np.histogram(de_loss_arr, density=True, bins='auto')
        # mid_bin = de_bins[:-1] + np.diff(de_bins) / 2
        # hist_mean = np.average(mid_bin, weights=de_hist)

        # print(f'{Fore.RED}HIST_MEAN:{Fore.GREEN}{fill_str} Arch {arch_str}'
        #       f' Comb DE Loss {hist_mean:.4f} {fill_str}{Style.RESET_ALL}')
        log_loss = {arch_str: [nb_nerons, nb_layers, act_fct, de_loss_avg, param_loss]}
        dat_frame = pd.DataFrame.from_dict(log_loss, orient='index')
        dat_frame.to_csv(f"DE_Loss_Logs/{self.session_ID}.csv", mode='a', header=False)
        print(dat_frame)

        del dat_frame, log_loss

        print(f'{Fore.RED}MEAN:{Fore.GREEN}{fill_str} Arch {arch_str}'
              f' Comb DE Loss {de_loss_avg:.4f} {fill_str}{Style.RESET_ALL}')

        return de_loss_avg

    def _make_param_map(self):
        """

            :return:
        """
        self.param_map = []
        for param_list in self.param_pos_vals:
            aux_linspace = np.linspace(0.0, 1.0, len(param_list) + 1)
            self.param_map.append(aux_linspace)

        return None

    def pick_arch_de(self, make_best=False):
        """
            Pick the appropriate architecture according to a differential evolution algorithm.

            :return:
        """
        net_bounds = [(0, 1) for _ in range(len(self.param_pos_vals))]

        result = differential_evolution(self._aux_min_func, net_bounds,
                                        maxiter=self.max_iter, disp=True
                                        )
        best_arch = self._conv_net_vec(result.x)
        if make_best:
            net_inst = DeepNetBonanza(best_arch[0], best_arch[1], train_dict=self.train_dict,
                                      drop_rate=0.2, name=self.name)
            net_inst.train_net(nb_epochs=self.epochs, mb_size=self.mb_size, save_model=True)

        return best_arch

    def get_de_loss(self, net_model):
        """
            Make the differential evolution loss.

            :return:
        """
        nb_params = count_params(net_model.trainable_weights)
        res_loss_fct = self.de_loss['Res_Loss']
        param_loss_fct = self.de_loss['Param_Loss']

        res_loss_val = res_loss_fct(self.train_dict['y'],
                                    net_model.predict(self.train_dict['x']).flatten()
                                    ).numpy()
        p_loss_val = param_loss_fct(nb_params)
        return res_loss_val, p_loss_val, {'Residual Loss': res_loss_val, 'Nb Params Loss': p_loss_val}


def main_test():
    """
        Test the differential evolution on fitting the ackley function
    :return:
    """

    def ackley_func_ndim(xNarray, aNorm=20, bNorm=0.2, cNorm=2 * np.pi):
        """
            Given a (M x N) array, the function computes the ackley function for the N dimension.
        """
        nDimm = xNarray.shape[1]
        sumSq = np.sum(np.square(xNarray), axis=1)
        cosSum = np.sum(np.cos(cNorm * xNarray), axis=1)

        ackFunc_T1 = - aNorm * np.exp(- bNorm * np.sqrt(sumSq / nDimm))
        ackFunc_T2 = - np.exp(cosSum / nDimm) + aNorm + np.exp(1)

        return ackFunc_T1 + ackFunc_T2

    # np.random.seed(0)
    #  ---------  Generate some random data and labels ---------
    nb_points = 5000
    a_low, b_high = -1, 1
    nb_dims = 2
    x_data = np.random.uniform(low=a_low, high=b_high, size=(nb_points, nb_dims))
    y_data = ackley_func_ndim(x_data)

    # --------- Initialise the differential evolution ---------
    max_iter = 3
    nb_epochs = 50
    train_dict = {'x': x_data, 'y': y_data}
    param_pos_vals = [
        [8, 16, 32, 64, 128, 256],  # Neuron nb
        list(range(1, 9)),  # Layers
        ['relu', 'elu', 'selu']
    ]
    de_inst = DiffEvolPickArch(train_dict, param_pos_vals, max_iter,
                               nb_epochs=nb_epochs, verbose=0, avg_nb=5)
    best_arch_ackley = de_inst.pick_arch_de()
    print(f'{Fore.BLUE}Finished DE algo, best {best_arch_ackley} architecture.')

    with open('DE_arch.pickle', 'wb') as pckl_out:
        pickle.dump(best_arch_ackley, pckl_out)

    return best_arch_ackley


if __name__ == "__main__":
    main_test()
