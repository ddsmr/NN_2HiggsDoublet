import numpy as np
from makeTrainNet import DeepNetBonanza
from tensorflow.keras.callbacks import Callback


class GetLastNSlope(Callback):

    def __init__(self, last_n=7, slope_toll=0.1):
        super(GetLastNSlope, self).__init__()
        self.last_n = last_n
        self.curr_losses = []
        self.slopes = []
        self.slope_toll = slope_toll
        self.slope_roll_avg = []

    def on_epoch_end(self, epoch, logs=None):
        curr_loss = logs.get('loss')
        self.curr_losses.append(curr_loss)
        self.curr_losses = self.curr_losses[-self.last_n:]

        if epoch >= self.last_n:
            y_arr = np.array(self.curr_losses)
            x_arr = np.array(range(self.last_n))
            fit_coeff = np.polyfit(x_arr, y_arr, 1)
            self.slopes.append(fit_coeff[0])

            slope_roll_avg = np.mean(self.slopes[-self.last_n:])
            self.slope_roll_avg.append(slope_roll_avg)

    def on_train_end(self, logs=None):
        from matplotlib import pyplot as plt
        plt.plot(self.slopes)
        plt.plot(self.slope_roll_avg)
        plt.show()


class NetTrainScaler:
    """
        Class to sort out when a certain network architecture has reached the learning plateau given a training budget
    """

    def __init__(self, nb_hidd_lay, nb_neur_lay, train_dict, mb_size=128, epoch_train_budget=3000):
        self.nb_hidd_lay = nb_hidd_lay
        self.nb_neur_lay = nb_neur_lay
        self.mb_size = mb_size
        self.epoch_train_budget = epoch_train_budget
        self.train_dict = train_dict

    def infer_plateau_scale(self):
        """
            Routine infers when the neural network has reached its potential by checking when the learning curve has
            hit a plateau.

            :return:
        """
        n_slope_callback = [GetLastNSlope()]
        net_inst = DeepNetBonanza(nb_layers=self.nb_hidd_lay,
                                  nb_neurs=self.nb_neur_lay, train_dict=self.train_dict, name='Ackley',
                                  drop_rate=0.2,
                                  verbose=1,
                                  )

        hist_dict = net_inst.train_net(nb_epochs=self.epoch_train_budget, mb_size=self.mb_size,
                                       save_model=False,
                                       show_train_plot=True,
                                       cust_callbacks=n_slope_callback
                                       )


def _main_test():
    """
        Auxiliary function to avoid name scope clashes
        :return:
    """
    #  ---------  Generate some random data and labels ---------
    nb_points = 1000
    a_low, b_high = -1, 1
    nb_dims = 2
    x_data = np.random.uniform(low=a_low, high=b_high, size=(nb_points, nb_dims))

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
    y_data = ackley_func_ndim(x_data)

    nb_hidd_lay = 2
    nb_neurs = 128
    train_budget = 3000

    train_dict = {'x': x_data, 'y': y_data}

    scale_inst = NetTrainScaler(nb_hidd_lay=nb_hidd_lay, nb_neur_lay=nb_neurs, train_dict=train_dict,
                                epoch_train_budget=train_budget)
    scale_inst.infer_plateau_scale()


if __name__ == "__main__":
    _main_test()

