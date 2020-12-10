# import json
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

###################### NEEDS REWRITE ######################
def load_files(data_dir='InitData/', header=None):
    """
        Load all the data files and merge them into one data frame.
    """
    if not bool(header):
        # header = ['Idx', 'mH1', 'mH2', 'mH3', 'tri_H3H2H1', 'tri_H3H1H1', 'tri_H3H2H2', 'eign_11', 'eign_21',
        #           'eign_31', 'lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s',
        #           'vev_v', 'vev_vs', 'vev_vss', 'Br_cascade', 'CrossX_Prod']
        header = ['mH1', 'mH2', 'mH3', 'trilinearH3H2H1', 'trilinearH3H3H3', 'trilinearH3H3H2', 'trilinearH3H3H1',
                  'trilinearH3H2H2', 'trilinearH3H1H1', 'trilinearH2H2H1', 'trilinearH2H1H1', 'trilinearH2H2H2',
                  'trilinearH1H1H1',
                  'rr_11', 'rr_21', 'rr_31', 'Totaldec', 'Totaldec2', 'Totaldec3', 'lambda_HS', 'lambda_sH',
                  'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s', 'v_s', 'v_ss', 'Br_cascade', 'crosx_topbr2',
                  'crossx_br_cascade_topbr'
                  ]
    from os.path import isfile, join
    file_names = [f_name for f_name in listdir(data_dir) if isfile(join(data_dir, f_name))]

    frames = []
    for file in file_names:
        data_frame = pd.read_csv(data_dir + file)
        data_frame.columns = header
        frames.append(data_frame)

    merged_df = pd.concat(frames, axis=0)
    return merged_df.sample(frac=1).reset_index(drop=True)


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
    return x_array[bool_mask, :], y_array[bool_mask]


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
            x_arr, y_arr = filter_data(x_arr_full, y_arr_full[:, attr_nb], perc_cut=filter_chisq)
        else:
            x_arr, y_arr = x_arr_full, y_arr_full[:, attr_nb]

        train_dict = {'x': x_arr, 'y': y_arr}

        if bool(test_arch) and isinstance(test_arch, dict):
            from Utils.makeTrainNet import DeepNetBonanza
            # nb_epochs, mb_size = test_arch['nb_epochs'], test_arch['mb_size']
            nb_neur, nb_lay = test_arch[analysis_attr][0], test_arch[analysis_attr][1]
            net_inst = DeepNetBonanza(nb_neur, nb_lay, train_dict, name=analysis_attr,
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
                        n_particles=10, nb_itters=100):
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

        new_arr = probe_contour(new_arr, list_probe, params_list, dat_bounder)
        print(new_arr.shape)
        new_points_arr = np.concatenate((new_points_arr, new_arr), axis=0)

    export_pred_data(new_points_arr, params_list)
    return None


if __name__ == "__main__":
    # np.random.seed(0)
    data_frame_m = load_files()
    # Filter by the 95% value of the chi squared measure.
    filter_by_chisq_m = 0.9
    norm_x_data = True
    # Attributes that go into chi square measure
    # chisq_attr_m = ['CrossX_Prod', 'mH3']
    chisq_attr_m = ['crossx_br_cascade_topbr', 'rr_11']
    arch_list = [[64, 4, 'relu'], [8, 6, 'relu']]
    arch_dict_test = {chi_att: arch_desc for chi_att, arch_desc in zip(chisq_attr_m, arch_list)}
    arch_dict_test.update({'nb_epochs': 3000, 'mb_size': 128})

    # Parameters of the model
    # model_params_m = ['lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s', 'vev_vs', 'vev_vss']
    model_params_m = ['lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s', 'v_s', 'v_ss']

    # import matplotlib.pyplot as plt
    # for attr in model_params_m:
    #     print(data_frame_m[attr].min(), data_frame_m[attr].max())
    #     data_frame_m.plot(use_index=True, y=attr)
    #     plt.show()

    bool_masks = [(-1, 1) for _ in range(len(model_params_m) - 2)]

    for min_max_tuple, cut_attr in zip(bool_masks, model_params_m):
        max_bound, min_bound = min_max_tuple[1], min_max_tuple[0]
        sub_bool_mask = (data_frame_m[cut_attr] > min_bound) & (max_bound > data_frame_m[cut_attr])
        data_frame_m = data_frame_m[sub_bool_mask]

    main_de(chisq_attr_m, model_params_m, data_frame_m,
            filter_chisq=filter_by_chisq_m,
            normalise_x_data=norm_x_data,
            n_samples=10000,
            mb_size=32,
            # test_arch=arch_dict_test,
            nb_epochs=300, averaging_number=10, max_iter=2
            )
    exit()
    # ------------------------------------------ Swarm: Predict new points ------------------------------------------
    # List of parameters to probe sub regions.
    # list_probe_m = ['lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s', 'v_s', 'v_ss']
    list_probe_m = ['v_s', 'v_ss']
    #  Use particle swarm to predict new points
    model_dict_m = {chisq_attr_m[0]: 'crossx_br_cascade_topbr_N64_L4_relu_linear_E3000_MB128',
                    chisq_attr_m[1]: 'rr_11_N8_L6_relu_linear_E3000_MB128'}

    swarm_loss_fct_list_m = [lambda x: 1 / np.exp(x),
                             lambda x: np.square(x - 1.0) / 0.2**2]

    nb_trial_points_m = 10000
    r_sigma_m = 0.3
    nb_seeds_m = int(nb_trial_points_m / 500)
    nb_sprouts_m = int(nb_trial_points_m / nb_seeds_m)
    main_part_swarm_min(data_frame_m,
                        list_probe_m, model_params_m, chisq_attr_m,
                        nb_seeds_m, nb_sprouts_m,
                        r_sigma_m,
                        model_dict_m,
                        swarm_loss_fct_list_m
                        )