import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import pyswarms as ps

from os import listdir
from colorama import Fore
from Utils.normArrays import ArrayNormaliser
from Utils.pickArchDe import DiffEvolPickArch
from Utils import smartRand as smartRnd
from Utils.pointBounder import DataBound

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_np_arr(model_params, chisq_list, data_frame, n_entries=0):
    """
        Load up the data frame and return the arrays for the model parameters to fit and the attributes that form the
        chi squared measure. Return the bounds for the parameters.
    """

    if bool(n_entries):
        data_frame = data_frame.sample(n=n_entries)

    data_bounds = {}
    for param in model_params:
        minP, maxP = data_frame[param].min(), data_frame[param].max()
        data_bounds[param] = {'Min': minP, 'Max': maxP}

    return {'x': data_frame[model_params].values, 'y': data_frame[chisq_list].values},\
        data_bounds, data_frame


def filter_data(x_array, y_array, perc_cut=0.99):
    """
        Apply cuts to the data, via cleaning from the yData. In this case take out everything that is above the 96% spread.
    """
    # y_array = y_array.values

    sorted_y = np.sort(y_array)
    cut_idx = int(perc_cut * y_array.shape[0])
    y_data_cut = sorted_y[cut_idx]

    bool_mask = y_array < y_data_cut
    return x_array[bool_mask, :], y_array[bool_mask], bool_mask


def split_test_train(x_array, y_array, split_frac=0.85):
    """
        Split into test and train arrays via the testSplit, for the data to fit and the fitting info.
    """
    # Random fraction of ~ (1 - testSplit) is allocated to the testing suite.
    bool_mask = np.array([True if np.random.uniform() < split_frac else False
                         for _ in range(x_array.shape[0])])

    x_arr_train, x_arr_test = x_array[bool_mask], x_array[np.logical_not(bool_mask)]
    y_arr_train, y_arr_test = y_array[bool_mask], y_array[np.logical_not(bool_mask)]

    return {'x': x_arr_train, 'y': y_arr_train},\
           {'x': x_arr_test, 'y': y_arr_test}


def norm_data_bounds(norm_inst, data_bound_dict, param_list, unpack_bound=False):
    """
        Normalises the bounds specified in the datBounds dict
    """

    min_list, max_list = [], []
    for param_nb, param in enumerate(param_list):
        p_min, p_max = data_bound_dict[param]['Min'], data_bound_dict[param]['Max']
        if unpack_bound:
            min_list.append(p_min)
            max_list.append(p_max)
        else:
            p_mean, p_std = norm_inst.normDict[str(param_nb)]['Mean'], norm_inst.normDict[str(param_nb)]['Std']
            min_list.append((p_min - p_mean) / p_std)
            max_list.append((p_max - p_mean) / p_std)

    swrm_bounds = (np.array(min_list), np.array(max_list))
    return swrm_bounds


def main_de(chisq_list, params_list, data_frame, filter_chisq=0.0, normalise_x_data=False, n_samples=0, test_arch=None,
            max_iter=2, averaging_number=50,
            nb_epochs=200,  mb_size=16,
            neur_nbs_list=None, hidd_lay_max=7, act_fct_list=None
            ):
    """
        Main def.
        :return:
    """
    data_dict, data_bounds, data_frame = get_np_arr(params_list, chisq_list, data_frame, n_entries=n_samples)
    x_arr_full, y_arr_full = data_dict['x'], data_dict['y']

    export_dict = {}

    # --- Create normaliser instance/ Normalise the bounds ----
    if normalise_x_data:
        norm_type = "MeanStdDimms"
        data_normaliser = ArrayNormaliser(x_arr_full, norm_type)
        x_arr_full = data_normaliser.normData(x_arr_full)
        export_dict.update({'NormInst': data_normaliser})
        data_bounds = norm_data_bounds(data_normaliser, data_bounds, params_list)
    else:
        data_bounds = norm_data_bounds(None, data_bounds, params_list, unpack_bound=True)

    arch_dict = {}
    # ---- Create and fit a neural network for each parameter in the chi2 list ----
    for attr_nb, analysis_attr in enumerate(chisq_list):
        if bool(filter_chisq):
            x_arr, y_arr, _ = filter_data(x_arr_full, y_arr_full[:, attr_nb], perc_cut=filter_chisq)
        else:
            x_arr, y_arr = x_arr_full, y_arr_full[:, attr_nb]

        train_dict = {'x': x_arr, 'y': y_arr}

        if bool(test_arch) and isinstance(test_arch, dict):
            from Utils.makeTrainNet import DeepNetBonanza
            # nb_epochs, mb_size = test_arch['nb_epochs'], test_arch['mb_size']
            nb_neur, nb_lay, act_fct = test_arch[analysis_attr]
            net_inst = DeepNetBonanza(nb_neurs=nb_neur, nb_layers=nb_lay, act_fct=act_fct,
                                      train_dict=train_dict,
                                      name=analysis_attr,
                                      drop_rate=0.2,
                                      verbose=1)
            net_inst.train_net(nb_epochs=nb_epochs, mb_size=mb_size, save_model=True, show_train_plot=True,
                               show_final_div_plot=True)
            continue

        # --------- Initialise the differential evolution ---------
        if neur_nbs_list is None:
            neur_nbs_list = [8, 16, 32, 64, 128, 256, 512, 1024]
        if act_fct_list is None:
            act_fct_list = ['relu', 'elu', 'selu']

        param_pos_vals = [
            neur_nbs_list,  # Neuron nb
            list(range(1, hidd_lay_max)),  # Layers
            act_fct_list  # Activation function
        ]
        de_inst = DiffEvolPickArch(train_dict, param_pos_vals, max_iter,
                                   nb_epochs=nb_epochs, mb_size=mb_size,
                                   verbose=0, avg_nb=averaging_number,
                                   name=analysis_attr)
        best_arch_ackley = de_inst.pick_arch_de(make_best=True)
        arch_dict.update({analysis_attr: best_arch_ackley})

        print(f'{Fore.BLUE}Finished DE algo, best {best_arch_ackley} architecture.')

    export_dict.update({'DataBounds': data_bounds, 'Archs': arch_dict})
    with open('DE_arch.pickle', 'wb') as pckl_out:
        pickle.dump(export_dict, pckl_out)


def _get_swarm_loss(pred_list, nb_part, swarm_loss_fct_list):
    """
        Construct the swarm loss from the swarm loss function list
        :param pred_list:
        :param nb_part:
        :param swarm_loss_fct_list:
        :return:
    """
    loss_arr = np.zeros(shape=(nb_part,))

    for loss_fct, pred_vals in zip(swarm_loss_fct_list, pred_list):
        pred_vals = pred_vals.flatten()
        loss_arr += loss_fct(pred_vals)

    return loss_arr


def _aux_func_min(x, net_dict, n_particles, chisq_list, swarm_loss_fct_list):
    """

    :param x:
    :return:
    """

    pred_dict = {}
    for chi_attr in net_dict.keys():
        attr_model = net_dict[chi_attr]
        pred_dict.update({chi_attr: attr_model.predict(x)})

    return _get_swarm_loss([pred_dict[chisq_attr] for chisq_attr in chisq_list], n_particles, swarm_loss_fct_list)


def _gen_points_around_seed(seed_vec, nb_points, r_sigma):

    rnd_eng = smartRnd.smartRand(None, None, None, None)
    param_dict = {'x' + str(i): seed_vec[i] for i in range(seed_vec.shape[0])}
    new_data_batch = np.array([seed_vec])

    nb_targeted = nb_points - 1
    for supra_epoch in range(nb_targeted):
        new_rnd_dict = rnd_eng.genRandUniform_Rn(param_dict, r_sigma)
        new_rnd_arr = np.array([new_rnd_dict['x' + str(i)] for i in range(seed_vec.shape[0])], ndmin=2)
        new_data_batch = np.concatenate((new_data_batch, new_rnd_arr), axis=0)

    return new_data_batch


def _form_pairwise(list_strs):
    """
        Forms pairs of the characters in listChar.
    """
    pair_list = []
    for nb_char1, char1 in enumerate(list_strs):
        for nb_char2 in range(nb_char1):
            pair = [char1, list_strs[nb_char2]]
            pair_list.append(pair)

    return pair_list


def probe_contour(sugg_arr, list_probe, model_params, dat_bounder):
    """

        :return:
    """
    all_bool = np.ones(shape=(sugg_arr.shape[0]), dtype=bool)
    param_probe_pairs = _form_pairwise(list_probe)

    # ---- Check the interiors of the contours
    for probe in param_probe_pairs:
        probe_a, probe_b = probe[0], probe[1]
        print(f'Probing {probe_a}/{probe_b} plane.')
        aidx, bidx = model_params.index(probe_a), model_params.index(probe_b)
        x_arr, y_arr = sugg_arr[:, aidx], sugg_arr[:, bidx]
        test_points = np.transpose(np.array([x_arr.flatten(), y_arr.flatten()]))

        pass_fail_dict = dat_bounder.inBounds(probe_a, probe_b, test_points)
        pass_points, fail_points, bool_mask_bounds = pass_fail_dict['Pass'], pass_fail_dict['Fail'], pass_fail_dict['BoolMask']

        all_bool = all_bool & bool_mask_bounds

    return sugg_arr[all_bool]


def export_pred_data(new_points_arr, header_list):
    """
        Exports the current set of points to the Predictions folder.

        :return:
    """
    new_data_frame = pd.DataFrame(data=new_points_arr, columns=header_list)
    new_data_frame.to_csv('Predictions/predPoints_wBounds_New.csv', index=False, sep=' ')


def main_part_swarm_min(data_frame, list_probe, params_list, chisq_list, nb_seeds, nb_sprouts, r_sigma, model_dict, swarm_loss_fct_list,
                        n_particles=10, nb_itters=100, test_param_cont=True, test_cust_consist=None):
    with open('DE_arch.pickle', 'rb') as pckl_in:
        run_dict = pickle.load(pckl_in)
    domain_bounds = run_dict['DataBounds']

    nb_dimms = len(params_list)
    net_dict = {}
    new_points_arr = np.transpose(np.array([[] for _ in range(nb_dimms)]))

    # _, _, data_frame = get_np_arr(params_list, chisq_list)
    dat_bounder = DataBound(data_frame)

    for model_name in model_dict.keys():
        model_path = 'Models/'
        model_path += model_dict[model_name]
        net_dict.update({model_name: tf.keras.models.load_model(model_path)})

    for try_nb in range(nb_seeds):
        print('-' * 90)
        # ---- Call instance of PSO ----
        ps_hyparams = {'c1': 0.1, 'c2': 0.1, 'w': 1.0, 'k': 4, 'p': 2}

        ps_optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                               dimensions=nb_dimms, options=ps_hyparams,
                                               bounds=domain_bounds
                                               )
        best_cost, best_vec = ps_optimizer.optimize(_aux_func_min,
                                                    iters=nb_itters, net_dict=net_dict, n_particles=n_particles,
                                                    chisq_list=chisq_list, swarm_loss_fct_list=swarm_loss_fct_list)

        # ---- Generate and normalise the new points if necessary
        new_arr = _gen_points_around_seed(best_vec, nb_sprouts, r_sigma)

        if 'NormInst' in run_dict.keys():
            data_normaliser = run_dict['NormInst']
            print('Set is normalised')
            new_arr = data_normaliser.invNormData(new_arr, None)
        else:
            data_normaliser = None

        if test_param_cont:
            new_arr = probe_contour(new_arr, list_probe, params_list, dat_bounder)
            print(new_arr.shape, ' Points remaining after probing conistency contours.')

        if bool(test_cust_consist) and isinstance(test_cust_consist, type(lambda x: 1)):
            new_arr = test_cust_consist(new_arr, params_list, net_dict, data_normaliser)

        new_points_arr = np.concatenate((new_points_arr, new_arr), axis=0)
        print('-' * 90)

    print(f'Total {new_points_arr.shape} new points')
    export_pred_data(new_points_arr, params_list)
    return None


def _test_func_ackley():
    """
        Subroutine to exemplify on the Ackley function test.
    :return:
    """

    def ackley_func_ndim(x_n_dimm_array, a_norm_fact=20, b_norm_fact=0.2, c_norm_fact=2 * np.pi):
        """
            Given a (M x N) array, the function computes the ackley function for the N dimension.
        """
        n_dimm = x_n_dimm_array.shape[1]
        sum_sq = np.sum(np.square(x_n_dimm_array), axis=1)
        cos_sum = np.sum(np.cos(c_norm_fact * x_n_dimm_array), axis=1)

        ack_func_t1 = - a_norm_fact * np.exp(- b_norm_fact * np.sqrt(sum_sq / n_dimm))
        ack_func_t2 = - np.exp(cos_sum / n_dimm) + a_norm_fact + np.exp(1)

        return ack_func_t1 + ack_func_t2

    def gen_ackley_df(nb_points=100, a_low=-1, b_high=1, nb_dims=2):
        """
            Generate the dataframe for the ackley function.
        :return:
        """
        # np.random.seed(0)
        #  ---------  Generate some random data and labels ---------

        x_data = np.random.uniform(low=a_low, high=b_high, size=(nb_points, nb_dims))
        y_data = ackley_func_ndim(x_data)

        param_list = ['x_n_' + str(x_dimm_count) for x_dimm_count in range(nb_dims)]
        chisq_list = ['y_Ackley']

        df_dict = {x_param: x_data[:, i] for x_param, i in zip(param_list, range(nb_dims))}
        df_dict.update({chisq_list[0]: y_data})
        ackley_df = pd.DataFrame.from_dict(df_dict)

        return param_list, chisq_list, ackley_df

    # --------- Initialise the differential evolution ---------
    param_list, chisq_attr, data_frame = gen_ackley_df()

    main_de(chisq_list=chisq_attr, params_list=param_list, data_frame=data_frame,
            # Optionals
            mb_size=32, nb_epochs=100,
            averaging_number=5, max_iter=1
            )

    model_list = ['y_Ackley_N512_L2_relu_linear_E10_MB32']
    list_probe = param_list
    nb_trial_points = 1000
    seed_red = 100
    r_sigma = 0.1
    nb_itters = 50

    model_dict = {attr: model_arch for attr, model_arch in zip(chisq_attr, model_list)}

    # Minimisation combination functions
    swarm_loss_fct_list = [lambda x: np.square(x - 0.0) / (0.01 ** 2)]
    nb_seeds = int(nb_trial_points / seed_red)
    nb_sprouts = int(nb_trial_points / nb_seeds)
    main_part_swarm_min(data_frame=data_frame, list_probe=list_probe, params_list=param_list,
                        chisq_list=chisq_attr, nb_seeds=nb_seeds, nb_sprouts=nb_sprouts, r_sigma=r_sigma,
                        model_dict=model_dict, swarm_loss_fct_list=swarm_loss_fct_list,
                        # Optionals
                        nb_itters=nb_itters,
                        test_param_cont=True
                        )


if __name__ == "__main__":
    print('Running Ackley function example')
    _test_func_ackley()