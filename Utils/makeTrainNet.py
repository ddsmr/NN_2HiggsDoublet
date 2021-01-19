import os
import types
# from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib

from dataclasses import dataclass
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback


from matplotlib.pyplot import rc
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
fontSizeGlobal = 16
plt.rc('font', size=fontSizeGlobal)
plt.rc('text', usetex=True)


def plot_training_hist(hist_dict, model_name, file_ext='pdf'):
    """

        :param model_name:
        :param file_ext:
        :param hist_dict:
        :return:
    """
    col_list = ['C0', 'C1', 'C2']
    label_dict = {'loss': r'$\mathrm{Loss}$',
                  'val_loss': r'$\mathrm{Val Loss}$',
                  'test_loss': r'$\mathrm{Test Loss}$'}
    train_hist = hist_dict['TrainHist']

    if 'TestHist' in hist_dict.keys():
        fig, axes = plt.subplots(ncols=1, nrows=2)
    else:
        fig, axes = plt.subplots(ncols=1, nrows=1)
        axes = [axes]

    for loss_nb, loss_type in enumerate(train_hist.history.keys()):
        loss_arr = train_hist.history[loss_type]
        axes[0].plot(loss_arr, label=label_dict[loss_type], c=col_list[loss_nb])
    axes[0].legend()

    if 'TestHist' in hist_dict.keys():
        test_hist = hist_dict['TestHist']
        test_loss_arr = test_hist.test_loss_list
        axes[1].plot(test_loss_arr, label=label_dict['test_loss'], c=col_list[-1])
        axes[1].legend()
    save_dir = f'Plots/{model_name}/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + f'TrainPlot.{file_ext}')
    plt.show()
    plt.close(fig)


def plot_div_plot(test_arr, pred_arr, model_name, aux_mod_str='', file_ext='png'):
    """

        :param file_ext:
        :param test_arr:
        :param pred_arr:
        :param model_name:
        :param aux_mod_str:
        :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(18.5, 10.5)
    point_nb_range = range(pred_arr.shape[0])
    resid_vals = (test_arr - pred_arr)

    axes[0].bar(point_nb_range, resid_vals, color='C0')
    axes[0].set_xlabel('Point Number')
    axes[0].set_ylabel(r'Residuals (Actual - Predicted)')

    axes[1].hist(resid_vals, color='C2')
    axes[1].set_xlabel('Actual - Predicted')

    save_dir = f'Plots/{model_name}/'
    os.makedirs(save_dir, exist_ok=True)

    if bool(aux_mod_str):
        aux_mod_str = '_' + aux_mod_str
    plt.savefig(save_dir + f'DivPlot' + aux_mod_str + f'.{file_ext}')
    plt.close(fig)
    # plt.show()
    return None


@dataclass
class TestEndEpoch(Callback):
    """
        Test the end epochs of a training.
    """
    test_dict: dict
    verbose: int = 1
    loss: str = 'mean_squared_error'

    def __post_init__(self):
        super(TestEndEpoch, self).__init__()

        self.test_loss_list = []
        if self.loss == 'mean_squared_error':
            self.loss_fct = tf.keras.losses.MeanSquaredError()

    def on_train_begin(self, logs=None):
        """
            Initialise the test loss list.
            :param logs:
            :return:
        """
        x_data, y_data = self.test_dict['x'], self.test_dict['y']
        model_pred = self.model.predict(x_data)
        self.test_loss_list.append(self.loss_fct(y_data, model_pred).numpy())

    def on_epoch_end(self, epoch, logs=None):
        x_data, y_data = self.test_dict['x'], self.test_dict['y']
        model_pred = self.model.predict(x_data)
        test_loss = self.loss_fct(y_data, model_pred).numpy()
        self.test_loss_list.append(test_loss)
        print(f'\nTest Loss at end of Epoch {epoch + 1}: {test_loss:.4f}')


# @dataclass
class PlotEndEpoch(Callback):
    """
        Test the end epochs of a training.
    """
    # plot_dict: dict
    # model_name: str
    # every_n_epoch: int = 1
    # verbose: int = 1
    # custom_plot_func = None

    def __init__(self, plot_dict, model_name, every_n_epoch=1, verbose=1, custom_plot_func=None, plot_div_plot=True):
        super(PlotEndEpoch, self).__init__()
        self.plot_dict = plot_dict
        self.model_name = model_name
        self.every_n_epoch = every_n_epoch
        self.verbose = verbose
        self.custom_plot_func = custom_plot_func
        self.plot_div_plot = plot_div_plot

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.every_n_epoch == 0 or epoch == 0:
            test_arr = self.plot_dict['TestArr']
            pred_arr = self.model.predict(self.plot_dict['PredArr']).flatten()
            if self.plot_div_plot:
                plot_div_plot(test_arr, pred_arr, self.model_name, aux_mod_str=str(epoch + 1))

            print('Plottting every n', bool(self.custom_plot_func))
            if bool(self.custom_plot_func):
                self.custom_plot_func(test_arr, pred_arr, self.plot_dict['PredArr'],
                                       self.model_name, aux_mod_str=str(epoch + 1),
                                       model_predictor=self.model.predict)


@dataclass
class DeepNetBonanza:
    """
        Test the bonanza.
    """
    nb_neurs: int
    nb_layers: int
    train_dict: dict
    drop_rate: float = 0.2
    act_fct: str = 'relu'
    out_lay_act_fct: str = 'linear'
    loss: str = 'mean_squared_error'
    optimizer: str = 'adam'
    test_dict: dict = None
    net_type: str = 'regression'
    verbose: int = 1
    valid_split: float = 0.0
    name: str = ''
    cust_plot_func: types.FunctionType = None

    def __post_init__(self):
        """
            Create the keras model.

            :return:
        """
        inp_dim = self.train_dict['x'].shape[1]
        if len(self.train_dict['y'].shape) == 2:
            out_dim = self.train_dict['y'].shape[1]
        else:
            out_dim = 1

        input_layer = layers.Input(shape=(inp_dim,))
        lay_list = [input_layer]
        for lay_nb in range(self.nb_layers):
            prev_lay = lay_list[-1]
            lay_list.append(layers.Dense(self.nb_neurs, activation=self.act_fct)(prev_lay))
            if bool(self.drop_rate):
                new_prev_lay = lay_list[-1]
                lay_list.append(layers.Dropout(self.drop_rate)(new_prev_lay))
        out_layer = layers.Dense(out_dim, activation=self.out_lay_act_fct)(lay_list[-1])

        # Specify the model
        model = Model(inputs=input_layer, outputs=out_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        self.model = model
        if self.verbose == 1:
            self.model.summary()

    def __str__(self):
        """
            Prints out the net info

            :return str:
        """
        print(self.model)
        net_str = f'Deep architecture with N: {self.nb_neurs} per layer with L: {self.nb_layers} hidden layers'

        if bool(self.drop_rate):
            net_str += f' w dropout: {self.drop_rate}.'
        net_str += f'\nAct Fct: {self.act_fct}\nFinal Layer Fct: {self.out_lay_act_fct}'
        return net_str

    def train_net(self, nb_epochs, mb_size, save_model=False, show_train_plot=False, show_final_div_plot=False,
                  plot_every_n=0, cust_callbacks=None, plot_div_plot=True):
        """

            :param save_model:
            :param nb_epochs:
            :param mb_size:
            :return:
        """
        arg_dict = {'x': self.train_dict['x'], 'y': self.train_dict['y'],
                    'batch_size': mb_size, 'epochs': nb_epochs,
                    'verbose': self.verbose}
        if bool(self.name):
            model_name = f'{self.name}_'
        else:
            model_name = ''
        model_name += f'N{self.nb_neurs}_L{self.nb_layers}_{self.act_fct}_{self.out_lay_act_fct}_'
        model_name += f'E{nb_epochs}_MB{mb_size}'

        if bool(cust_callbacks):
            callbacks = cust_callbacks
        else:
            callbacks = []

        hist_dict = {}
        if bool(self.valid_split):
            arg_dict.update({'validation_split': self.valid_split})
        if bool(self.test_dict):
            end_epoch_call = TestEndEpoch(self.test_dict, verbose=self.verbose, loss=self.loss)
            callbacks.append(end_epoch_call)
            hist_dict.update({'TestHist': end_epoch_call})

        if bool(plot_every_n):
            if bool(self.test_dict) is False:
                test_arr = arg_dict['y'].flatten()
                pred_arr = arg_dict['x']

            else:
                test_arr = self.test_dict['y'].flatten()
                pred_arr = self.test_dict['x']

            plot_dict = {'TestArr': test_arr, 'PredArr': pred_arr}
            plot_end_epoch = PlotEndEpoch(plot_dict, model_name, every_n_epoch=plot_every_n,
                                          plot_div_plot=plot_div_plot,
                                          custom_plot_func=self.cust_plot_func)
            callbacks.append(plot_end_epoch)

        if bool(callbacks):
            arg_dict.update({'callbacks': callbacks})

        # Fit the model
        train_hist = self.model.fit(**arg_dict)
        hist_dict.update({'TrainHist': train_hist})

        if show_train_plot:
            plot_training_hist(hist_dict, model_name)
        if show_final_div_plot:
            if bool(self.test_dict) is False:
                test_arr = arg_dict['y'].flatten()
                pred_arr = self.model.predict(arg_dict['x']).flatten()
            else:
                test_arr = self.test_dict['y'].flatten()
                pred_arr = self.model.predict(self.test_dict['x']).flatten()
            plot_div_plot(test_arr, pred_arr, model_name, aux_mod_str='Final')

        if save_model:
            self.model.save('Models/' + model_name)
        return train_hist


def _main_test():

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

    def cust_plot(y_true_arr, y_pred_arr, x_arr, mod_str, aux_mod_str, model_predictor=None):
        """
            Custom plot to plot every n.
        :return:
        """
        # fig = plt.figure(figsize=(20, 9))
        # (ax1, ax2) = fig.subplots(nrows=1, ncols=2)
        # cmap = 'RdYlBu'

        # plot1 = ax1.scatter(x=x_arr[:, 0], y=x_arr[:, 1], c=y_true_arr, cmap=cmap)
        # fig.colorbar(plot1, ax=ax1, label=r"$f_{\mathrm{Ack - True}}(x_1, x_2)$", shrink=0.7,
        #              format='%.2f')
        # ax1.set_title('True')
        # ax1.set_aspect('equal', adjustable='box')
        #
        # plot2 = ax2.scatter(x=x_arr[:, 0], y=x_arr[:, 1], c=y_pred_arr, cmap=cmap)
        # fig.colorbar(plot2, ax=ax2, label=r"$f_{\mathrm{Ack - Pred}}(x_1, x_2)$", shrink=0.7)
        #
        # ax2.set_title('Predicted')
        # ax2.set_aspect('equal', adjustable='box')

        epoch = aux_mod_str
        gran_nb = 500
        x1_lin_space = np.linspace(a_low, b_high, num=gran_nb)
        x2_lin_space = np.linspace(a_low, b_high, num=gran_nb)
        X1, X2 = np.meshgrid(x1_lin_space, x2_lin_space)

        net_vec_array = []
        for i in range(gran_nb):
            for j in range(gran_nb):
                # coord_vec = np.array([X1[i, j], X2[j, i]])
                coord_vec = [X1[i, j], X2[i, j]]
                net_vec_array.append(coord_vec)
        net_vec_array = np.array(net_vec_array).astype('float64')

        z_flat = model_predictor(net_vec_array)

        Z = z_flat.reshape((gran_nb, gran_nb))
        # print(z_flat.shape, Z.shape)

        cMap = 'RdYlBu'
        # cMap = 'coolwarm_r'
        fig = plt.figure(figsize=(12, 9))
        (ax, ax2) = fig.subplots(1, 2)

        xArr, yArr = x_arr, y_true_arr
        # surfPlot = ax2.scatter(xArr[:, 0], xArr[:, 1], c=yArr, cmap=cMap)

        z2_flat = ackley_func_ndim(net_vec_array)
        Z2 = z2_flat.reshape((gran_nb, gran_nb))
        surfPlot2 = ax2.contourf(X1, X2, Z2, cmap=cMap,
                                 levels=11,
                                 vmin=0, vmax=5)
        # fig.colorbar(surfPlot2, ax=ax2, label=r"$f_{\mathrm{Ack - Act}}(x_1, x_2)$", shrink=0.5)
        ax2.set_xlabel(r"$x_1$")
        # ax2.set_ylabel(r"$x_2$")
        ax2.set_title("Original Function")
        ax2.set_aspect('equal', adjustable='box')

        surfPlot = ax.contourf(X1, X2, Z, cmap=cMap,
                               levels=11,
                               vmin=0, vmax=5)
        # fig.colorbar(surfPlot, ax=ax, label=r"$f_{\mathrm{Ack - Pred}}(x_1, x_2)$", shrink=0.5,
        #              format='%.2f')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title("NN Inferred Function")
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(surfPlot2, ax=[ax, ax2], label=r"$f(x_1, x_2)$", location='bottom',
                     format='%.2f')
        # surfPlot3 = ax3.contourf(X1, X2, np.abs(Z2 - Z), cmap='gray_r', levels=15)
        # fig.colorbar(surfPlot3, ax=ax3, label=r"$\mathrm{Res}$", shrink=0.5)
        # ax3.set_xlabel(r"$x_1$")
        # ax3.set_ylabel(r"$x_2$")
        # ax3.set_title("Absolute residuals")
        # ax3.set_aspect('equal', adjustable='box')

        fig.suptitle(f"NetArch: N{nb_neurs} L{nb_hidd_lay} ({act_fct})  ---  At end of Training Epoch:{epoch}")
        # plt.tight_layout()

        plt_dir = f'Plots/Ackley_N{nb_neurs}_L{nb_hidd_lay}_M32/'
        plt_str = f'MeshPlots_E{epoch}.png'
        pathlib.Path(plt_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(f'{plt_dir}{plt_str}', bbox_inches='tight', pad_inches=0.1)

        # plt.show()
        plt.close(fig)

    # np.random.seed(0)
    #  ---------  Generate some random data and labels ---------
    nb_points = 5000
    a_low, b_high = -1, 1
    nb_dims = 2
    x_data = np.random.uniform(low=a_low, high=b_high, size=(nb_points, nb_dims))
    y_data = ackley_func_ndim(x_data)

    train_dict = {'x': x_data, 'y': y_data}

    # nb_neurs = 128
    # nb_hidd_lay = 3

    arch_list = [[16, 4, 'elu'], [32, 4, 'elu'], [64, 3, 'elu'], [128, 3, 'relu'], [256, 2, 'relu'],
                 [512, 1, 'relu'],
                 [512, 2, 'elu']]

    for arch in arch_list:
        nb_neurs, nb_hidd_lay, act_fct = arch
        net_inst = DeepNetBonanza(nb_neurs, nb_hidd_lay, train_dict, act_fct=act_fct, name='Ackley',
                                  #  test_dict={'x': np.random.uniform(size=(100, 2)),
                                  #            'y': np.random.uniform(size=(100, 1))},
                                  drop_rate=0.2,
                                  verbose=1,
                                  cust_plot_func=cust_plot
                                  # valid_split=0.2
                                  )
        print(net_inst)
        hist_dict = net_inst.train_net(nb_epochs=300, mb_size=32, save_model=False, show_train_plot=False,
                                       plot_every_n=25, plot_div_plot=False
                                       )
        # del net_inst


if __name__ == "__main__":
    _main_test()

