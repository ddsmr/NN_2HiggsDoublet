# import json
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import pyswarms as ps

from os import listdir
from colorama import Fore
from Utils.normArrays import ArrayNormaliser
from pickArchDe import DiffEvolPickArch
from Utils import smartRand as smartRnd
from Utils.pointBounder import DataBound

###################### NEEDS REWRITE ######################
def load_files(data_dir='InitData/', header=None):
    """
        Load all the data files and merge them into one data frame.
    """
    if not bool(header):
        header = ['Idx', 'mH1', 'mH2', 'mH3', 'tri_H3H2H1', 'tri_H3H1H1', 'tri_H3H2H2', 'eign_11', 'eign_21',
                  'eign_31', 'lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s',
                  'vev_v', 'vev_vs', 'vev_vss', 'Br_cascade', 'CrossX_Prod']

    file_names = [f_name for f_name in listdir(data_dir)]

    frames = []
    for file in file_names:
        data_frame = pd.read_csv(data_dir + file)
        data_frame.columns = header
        frames.append(data_frame)

    merged_df = pd.concat(frames, axis=0)
    return merged_df.sample(frac=1).reset_index(drop=True)


def get_np_arr(model_params, chisq_list):
    """
        Load up the data frame and return the arrays for the model parameters to fit and the attributes that form the
        chi squared measure. Return the bounds for the parameters.
    """
    data_frame = load_files()

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


def norm_data_bounds(norm_inst, data_bound_dict, param_list):
    """
        Normalises the bounds specified in the datBounds dict
    """

    min_list, max_list = [], []
    for param_nb, param in enumerate(param_list):
        p_min, p_max = data_bound_dict[param]['Min'], data_bound_dict[param]['Max']
        p_mean, p_std = norm_inst.normDict[str(param_nb)]['Mean'], norm_inst.normDict[str(param_nb)]['Std']
        min_list.append((p_min - p_mean) / p_std)
        max_list.append((p_max - p_mean) / p_std)

    swrm_bounds = (np.array(min_list), np.array(max_list))
    return swrm_bounds


def main_de(chisq_list, params_list, filter_chisq=0.0, normalise_x_data=False):
    """
        Main def.
        :return:
    """
    data_dict, data_bounds, data_frame = get_np_arr(params_list, chisq_attr_m)
    x_arr_full, y_arr_full = data_dict['x'], data_dict['y']
    export_dict = {}

    # --- Create normaliser instance/ Normalise the bounds ----
    if normalise_x_data:
        norm_type = "MeanStdDimms"
        data_normaliser = ArrayNormaliser(x_arr_full, norm_type)
        data_bounds = norm_data_bounds(data_normaliser, data_bounds, params_list)
        x_arr_full = data_normaliser.normData(x_arr_full)
        export_dict.update({'NormInst': data_normaliser})

    arch_dict = {}
    # ---- Create and fit a neural network for each parameter in the chi2 list ----
    for attr_nb, analysis_attr in enumerate(chisq_list):
        if bool(filter_chisq):
            x_arr, y_arr = filter_data(x_arr_full, y_arr_full[:, attr_nb], perc_cut=filter_chisq)
        else:
            x_arr, y_arr = x_arr_full, y_arr_full[:, attr_nb]

        train_dict = {'x': x_arr, 'y': y_arr}

        # --------- Initialise the differential evolution ---------
        max_iter = 1
        nb_epochs = 2
        averaging_number = 1
        param_pos_vals = [
            [8, 16],  # Neuron nb
            list(range(1, 3)),  # Layers
            ['relu', ]
        ]
        de_inst = DiffEvolPickArch(train_dict, param_pos_vals, max_iter,
                                   nb_epochs=nb_epochs, verbose=0, avg_nb=averaging_number,
                                   name=analysis_attr)
        best_arch_ackley = de_inst.pick_arch_de(make_best=True)
        arch_dict.update({analysis_attr: best_arch_ackley})

        print(f'{Fore.BLUE}Finished DE algo, best {best_arch_ackley} architecture.')

    export_dict.update({'DataBounds': data_bounds, 'Archs': arch_dict})
    with open('DE_arch.pickle', 'wb') as pckl_out:
        pickle.dump(export_dict, pckl_out)

###################### NEEDS REWRITE ######################
def _get_swarm_loss(pred_list, nb_part, sigma_H=3.0):
    # print(pred_dict)
    # print(pred_dict['mH3'].shape, np.square(pred_dict['mH3'] - 125.09))
    loss_arr = np.zeros(shape=(nb_part,))

    pred_vals_sigma = pred_list[0].flatten()
    pred_vals_H = pred_list[1].flatten()
    loss_arr += np.square(pred_vals_H - 125.09) / sigma_H**2
    loss_arr += 1 / np.exp(pred_vals_sigma)

    return loss_arr


def _aux_func_min(x, net_dict, n_particles, chisq_list):
    """

    :param x:
    :return:
    """

    pred_dict = {}
    for chi_attr in net_dict.keys():
        attr_model = net_dict[chi_attr]
        pred_dict.update({chi_attr: attr_model.predict(x)})

    return _get_swarm_loss([pred_dict[chisq_attr] for chisq_attr in chisq_list], n_particles)


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


def main_part_swarm_min(list_probe, params_list, chisq_list, nb_seeds, nb_sprouts, nb_dimms, r_sigma, model_dict,
                        n_particles=10, nb_itters=100):
    with open('DE_arch.pickle', 'rb') as pckl_in:
        run_dict = pickle.load(pckl_in)
    domain_bounds = run_dict['DataBounds']

    net_dict = {}
    new_points_arr = np.transpose(np.array([[] for _ in range(nb_dimms)]))

    _, _, data_frame = get_np_arr(params_list, chisq_list)
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
                                               bounds=domain_bounds)
        best_cost, best_vec = ps_optimizer.optimize(_aux_func_min,
                                                    iters=nb_itters, net_dict=net_dict, n_particles=n_particles,
                                                    chisq_list=chisq_list)

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

    # Filter by the 95% value of the chi squared measure.
    filter_by_chisq_m = 0.95
    norm_x_data = True
    # Attributes that go into chi square measure
    chisq_attr_m = ['CrossX_Prod', 'mH3']
    # Parameters of the model
    model_params_m = ['lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s', 'vev_vs', 'vev_vss']
    main_de(chisq_attr_m, model_params_m, filter_by_chisq_m, normalise_x_data=norm_x_data)

    # ------------------------------------------ Swarm: Predict new points ------------------------------------------
    # List of parameters to probe sub regions.
    list_probe_m = ['vev_vs', 'vev_vss', 'lambda_HS', 'lambda_sH', 'lambda_sS', 'lambda_H', 'lambda_S', 'lambda_s']
    #  Use particle swarm to predict new points
    model_dict_m = {chisq_attr_m[0]: 'CrossX_Prod_N8_L1_relu_linear_E2_MB128',
                    chisq_attr_m[1]: 'mH3_N16_L2_relu_linear_E2_MB128'}
    nb_trial_points_m = 1000
    r_sigma_m = 0.3
    nb_seeds_m = int(nb_trial_points_m / 500)
    nb_sprouts_m = int(nb_trial_points_m / nb_seeds_m)
    main_part_swarm_min(list_probe_m, model_params_m, chisq_attr_m, nb_seeds_m, nb_sprouts_m,
                        len(model_params_m), r_sigma_m,
                        model_dict_m)