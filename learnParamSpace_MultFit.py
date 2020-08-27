import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
import json
import pickle
import random
from matplotlib import rc

# tensorflow stuffs.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

import sys
sys.path.insert(1, '../Utils')
from printUtils import *
import smartRand as smrnd
import os
import subprocess
import matplotlib as mpl
FNULL = open(os.devnull, 'w')
mpl.style.use('seaborn')

# Pretty python traceback outputs
try:
    import colored_traceback
    colored_traceback.add_hook()
except Exception as e:
    print(e)


def plotSlice(xData, yOrign, yPred, slices=[0, 1]):
    '''
        Plots the slice of the fit of the data.
    '''
    nbPoints = xData.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    origFig = ax1.scatter(xData[:, slices[0]], xData[:, slices[1]], c=yOrign.reshape(nbPoints,), marker='o',
                          cmap='tab20c')
    cbarOrig = fig.colorbar(origFig, ax=ax1)
    ax1.set_title('Original Values')

    predFig = ax2.scatter(xData[:, slices[0]], xData[:, slices[1]], c=yPred.reshape(nbPoints,), marker='o',
                          cmap='tab20c')
    ax2.set_title('Prdicted Values')
    cbarPred = fig.colorbar(predFig, ax=ax2)

    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    plt.savefig('fittedData.pdf')
    plt.show()

    return None


def getTopNVal(dataArray, nVal=1.0):
    '''
        Return the top n = nVal % value corresponding to their values (default at 10%).
        dataArray must be a (N, ) shaped array.
    '''
    nbOfVals = dataArray.shape[0]
    return np.sort(dataArray)[int((nVal / 100) * nbOfVals)]


def bestChi2_sampling(nbPoints, NN_model, runCard_dict):
    '''
        Returns nbPoints of random points suggested by the model, which is specified by the runCard_dict.
    '''
    aLow, bHigh, shape = runCard_dict['Bounds'][0], runCard_dict['Bounds'][1], runCard_dict['Shape']
    nbPts_f, topN = runCard_dict['rndFrac'], runCard_dict['topN']

    newPts = 0
    newDataBatch = np.transpose(np.array([[] for _ in range(nbDimms)]))

    while newPts < int(nbPts_f * nbPoints):
        #  Predict the new data from the candidate
        xDataCand = np.random.uniform(low=aLow, high=bHigh, size=(shape[0], shape[1]))
        yPred = model.predict(xDataCand).reshape(shape[0],)
        chiSq_Array = getChi2(yPred)
        topChiVal = getTopNVal(chiSq_Array, nVal=topN)

        chiBool = chiSq_Array < topChiVal
        xDataNew = xDataCand[chiBool]
        newDataBatch = np.concatenate((newDataBatch, xDataNew), axis=0)

        newPts += xDataNew.shape[0]
        # print(newPts)

    xDataFill = np.random.uniform(low=aLow, high=bHigh, size=(int((1 - nbPts_f) * nbPoints), shape[1]))
    newDataBatch = np.concatenate((newDataBatch, xDataFill), axis=0)

    # plt.scatter(newDataBatch[:, 0], newDataBatch[:, 1], c='r')
    # plt.scatter(xDataFill[:, 0], xDataFill[:, 1], c='b')
    # plt.show()
    # exit()
    #  np.concatenate((newDataBatch, xData), axis=0)
    return newDataBatch


def swarmRndSeed_sampling(nbPoints, model, runCard_dict, n_particles=10, nbIters=100):
    '''
        The function requires a neural net model, which is minimised via the particle swarm. Once the particle swarm
        converges to a point, the point is used as a seed which then generates points around in the hypersphere around
        it determined by the rSigma parameter. The number of Points is split between the seeded algorithm and some
        purely random points.
    '''

    shape = runCard_dict['Shape']
    nbDimms, rSigma = runCard_dict['nbDimms'], runCard_dict['rSigma']

    # Set up hyperparams if passed, otherwise Initialise via specified method
    if 'hyperParams' in runCard_dict.keys():
        hyperParams = runCard_dict['hyperParams']
    else:
        hyperParams = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 5, 'p': 2}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=nbDimms, options=hyperParams)

    # Perform optimization; extract the bes =t cist and its posision
    cost, xVec_seed = optimizer.optimize(model.predict, iters=nbIters)

    # Initialise random seeder
    rndEng = smrnd.smartRand(None, None, None, None)
    paramDict = {'x' + str(i): xVec_seed[i] for i in range(xVec_seed.shape[0])}
    newDataBatch = np.array([xVec_seed])

    nbTargeted = nbPoints - 1
    for supraEpoch in range(nbTargeted):
        newRndDict = rndEng.genRandUniform_Rn(paramDict, rSigma)
        newRndArr = np.array([newRndDict['x' + str(i)] for i in range(xVec_seed.shape[0])], ndmin=2)
        newDataBatch = np.concatenate((newDataBatch, newRndArr), axis=0)

    return newDataBatch, np.array([xVec_seed])


def NN_suggestPoints(nbPoints, model, mthd, runCard_dict):
    '''
        The neural net 'model' suggest a number of nbPoints, via the mthd, with the runCard_dict set of params.
    '''

    if mthd == 'bestChi2':
        return bestChi2_sampling(nbPoints, model, runCard_dict)
    elif mthd == 'swarmRndSeed':
        return swarmRndSeed_sampling(nbPoints, model, runCard_dict)


def getNumpyArrays(fileHandle, attrList):
    '''
        Open the json file with the parameter space points and their corresponding χ² values. Return the data as
        numpy arrays.
    '''
    import csv

    paramList = ['p_L1', 'p_L2', 'p_L3', 'p_L4', 'p_L5', 'p_L6', 'p_L7', 'p_L8', 'p_u11sq', 'p_u22sq', 'p_m12sq',
                 'p_ussq']

    with open('TrainingData/' + fileHandle,  newline='', encoding='utf-8') as f:
        csvReader = csv.reader(f)
        headRow = csvReader.__next__()[0]
        delimChar = headRow[5]
        headRow = headRow.split(delimChar)

        # Get the header position for the parameters and the model attributes.
        paramPos, attrPos = [], []
        for param in paramList:
            headPos = headRow.index(param)
            paramPos.append(headPos)

        for attr in attrList:
            headPos = headRow.index(attr)
            attrPos.append(headPos)

        xVecList, yVecList = [], []
        for row in csvReader:
            xVec = [float(row[0].split(delimChar)[pos]) for pos in paramPos]
            yVec = [float(row[0].split(delimChar)[pos]) for pos in attrPos]
            xVecList.append(xVec), yVecList.append(yVec)

    return {'xData': np.array(xVecList), 'yData': np.array(yVecList), 'ParamList': paramList}


def initArch(nbOfDimms, y_dimm, nbNeur, actFct, λregVal=0.0):
    '''
        Initialises the linear architecture used to fit the parameter space.
    '''
    # Initialise the Neural network
    inputLayer = keras.Input(shape=(nbOfDimms,))
    denseLayer = layers.Dense(nbNeur, activation=actFct, activity_regularizer=regularizers.l2(λregVal))
    a1 = denseLayer(inputLayer)
    outLayer = layers.Dense(y_dimm, activation=actFct, activity_regularizer=regularizers.l2(λregVal))(a1)
    model = keras.Model(inputs=inputLayer, outputs=outLayer, name="Learn_HosotaniSO11")

    #  Plot out the neural net and compile it
    # keras.utils.plot_model(model, "FitModel/Learn_HosotaniSO11.png", show_shapes=True)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Save the initial random weights, used for reinitialisation.
    Wsave = model.get_weights()

    return model, Wsave


def makeLinArch(inpDim, nbNeurons, nbLayers, outDim, actFct='relu',
                lossFct='mean_squared_error', optType='adam', dropRate=0.2):
    '''
        Make a linear fully connected architecture with nbNeurons per layer, where the intemediate number of layers is
        specified by nbLayers, the dimension of the input and the output layers are specified by inpDim and outDim.
    '''
    # Start offf with the layer
    inputLayer = keras.Input(shape=(inpDim,))

    layerDict = {}
    dropDict = {}
    for layerNb in range(nbLayers):
        layerDict[str(layerNb)] = layers.Dense(nbNeurons, activation=actFct)
        dropDict[str(layerNb)] = layers.Dropout(dropRate)

    layerDict['0'] = layerDict['0'](inputLayer)
    dropDict['0'] = dropDict['0'](layerDict['0'])
    for layerNb in range(nbLayers - 1):
        # layerDict[str(layerNb + 1)] = layerDict[str(layerNb + 1)](layerDict[str(layerNb)])
        layerDict[str(layerNb + 1)] = layerDict[str(layerNb + 1)](dropDict[str(layerNb)])
        dropDict[str(layerNb + 1)] = dropDict[str(layerNb + 1)](layerDict[str(layerNb)])

    # outLayer = layers.Dense(outDim, activation=actFct)(layerDict[str(nbLayers - 1)])
    outLayer = layers.Dense(outDim, activation=actFct)(dropDict[str(nbLayers - 1)])

    model = keras.Model(inputs=inputLayer, outputs=outLayer)
    model.compile(loss=lossFct, optimizer=optType, metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    model.summary()
    # keras.utils.plot_model(model, fitHandle + ".png", show_shapes=True)

    return model

class ArrayNorms:
    '''
        Normalisation class, has the direct and the inverse transformations included as methods.
    '''

    def __init__(self, datArray):
        '''
            Initialise the normalisation class with the array values.
        '''
        # self.dataArray = datArray
        self.normDict = {}

        if len(datArray.shape) == 2:
            nbAttrs = datArray.shape[1]

            for attNb in range(nbAttrs):
                attMean = np.mean(datArray[:, attNb])
                attStd = np.std(datArray[:, attNb])
                self.normDict[str(attNb)] = {'Mean': attMean, 'Std': attStd}

        elif len(datArray.shape) == 1:
            attMean = np.mean(datArray)
            attStd = np.std(datArray)
            self.normDict['0'] = {'Mean': attMean, 'Std': attStd}

    def normData(self, datArray):
        '''
            Normalise the data via Z = (X - μ) / σ, where this is done for each dimension.
        '''
        # if datArray is None:
        #     datArray = self.dataArray
        normDict = {}
        normArray = []

        if len(datArray.shape) == 2:
            nbAttrs = datArray.shape[1]

            for attNb in range(nbAttrs):
                attMean = self.normDict[str(attNb)]['Mean']
                attStd = self.normDict[str(attNb)]['Std']

                normAttArr = (datArray[:, attNb] - attMean) / attStd
                normArray.append(normAttArr)
                # normDict[str(attNb)] = {'Mean': attMean, 'Std': attStd}
                # print(f'Att nb {attNb}:', np.mean(normAttArr[:]), np.std(normAttArr[:]))
            normArray = np.transpose(np.array(normArray))

        elif len(datArray.shape) == 1:
            attMean = self.normDict['0']['Mean']
            attStd = self.normDict['0']['Std']
            normAttArr = (datArray - attMean) / attStd
            normArray = np.array(normAttArr)
            # normDict['0'] = {'Mean': attMean, 'Std': attStd}

        # self.normDict = normDict
        # return normArray, normDict
        return normArray

    def invScaleData(self, datArray):
        '''
            Given the transformation dictionary, the function performs the inverse transformation to normData.
        '''
        transDict = self.normDict
        if transDict is None:
            print('Class instance does not have a direct transformation dictionary. Run normData() on the original set.')
            raise

        if len(datArray.shape) == 2:
            nbAttrs = datArray.shape[1]
            normArray = []

            for attrNb in range(nbAttrs):
                attrMean, attrStd = transDict[str(attrNb)]['Mean'], transDict[str(attrNb)]['Std']
                normAttArr = datArray[:, attrNb] * attrStd + attrMean
                normArray.append(normAttArr)
            normArray = np.transpose(np.array(normArray))

        elif len(datArray.shape) == 1:
            attrMean, attrStd = transDict['0']['Mean'], transDict['0']['Std']
            normArray = datArray[:, ] * attrStd + attrMean

        return normArray


def normData(datArray):
    '''
        Normalise the data via Z = (X - μ) / σ, where this is done for each dimension.
    '''
    normDict = {}
    normArray = []

    if len(datArray.shape) == 2:
        nbAttrs = datArray.shape[1]

        for attNb in range(nbAttrs):
            attMean = np.mean(datArray[:, attNb])
            attStd = np.std(datArray[:, attNb])

            normAttArr = (datArray[:, attNb] - attMean) / attStd
            normArray.append(normAttArr)
            normDict[str(attNb)] = {'Mean': attMean, 'Std': attStd}

            # print(f'Att nb {attNb}:', np.mean(normAttArr[:]), np.std(normAttArr[:]))
        normArray = np.transpose(np.array(normArray))

    elif len(datArray.shape) == 1:
        attMean = np.mean(datArray)
        attStd = np.std(datArray)
        normAttArr = (datArray - attMean) / attStd
        normArray = np.array(normAttArr)
        normDict['0'] = {'Mean': attMean, 'Std': attStd}

    print(normArray.shape)
    pp(normDict)
    return normArray, normDict


def invScaleData(datArray, transDict):
    '''
        Given the transformation dictionary, the function performs the inverse transformation to normData.
    '''
    if len(datArray.shape) == 2:
        nbAttrs = datArray.shape[1]
        normArray = []

        for attrNb in range(nbAttrs):
            attrMean, attrStd = transDict[str(attrNb)]['Mean'], transDict[str(attrNb)]['Std']
            normAttArr = datArray[:, attrNb] * attrStd + attrMean
            normArray.append(normAttArr)
        normArray = np.transpose(np.array(normArray))

    elif len(datArray.shape) == 1:
        attrMean, attrStd = transDict['0']['Mean'], transDict['0']['Std']
        normArray = datArray[:, ] * attrStd + attrMean

    return normArray


def genPointID(threadNumber):
    '''
        Generates a random point ID
    '''
    currTime = strftime("-%d%m%Y%H%M%S", gmtime())
    pointKey = 'Point T' + str(threadNumber) + "-" + str(int(random.uniform(1, 1000))) + currTime

    return pointKey


def formChi2Array(yData, chi2Dict, attrList):
    '''
        Given the label array and the central values, standard deviations of the labels, function forms a combined χ²
        value.
    '''
    chi2Array = np.zeros((yData.shape[0],))
    for attrNb, attr in enumerate(attrList):
        centralVal, stdVal = chi2Dict[attr]['CentralVal'], chi2Dict[attr]['StD']
        indvChi2 = (yData[:, attrNb] - centralVal) ** 2 / stdVal**2
        chi2Array = chi2Array + indvChi2
    # plt.hist(chi2Array)
    # plt.show()
    return chi2Array


def setMnStd(dataArray, attrList, f_stdVal=0.05, showPlts=False):
    '''
        Get the mean of the atributes specified in the attribute list. Set the Std val specified for stdVal as f%
        of the mean for the CHI2 calculation. Return Dictionary of the attributes.
    '''
    attrDict = {attr: {'CentralVal': None, 'StD': None} for attr in attrList}
    for pltNb in range(len(attrDict)):
        centralVal = np.mean(dataArray[:, pltNb])
        stdVal = f_stdVal * centralVal
        attrDict[attrList[pltNb]]['CentralVal'], attrDict[attrList[pltNb]]['StD'] = centralVal, stdVal

        if showPlts:
            plt.hist(dataDict['yData'][:, pltNb])
            plt.title(attrList[pltNb])
            plt.show()

    return attrDict


def showStats(fitModel, history, testData_norm, yDataTest, attrList, saveStr):
    '''
        Show statistics related to the fitting.
    '''
    saveDir = 'AnalysisPlots/MultiDim/' + saveStr + '/'
    subprocess.call('mkdir ' + saveDir, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    # Show the training losses
    plt.plot(history.history['loss'], c='C0')
    plt.plot(history.history['val_loss'], c='C1')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(saveDir + 'trainLosses.pdf')
    plt.show()

    # Plot the predictions and original histograms
    modelPred = fitModel.predict(testData_norm)

    for predNb in range(modelPred.shape[1]):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.suptitle(f'Attribute Distibution {attrList[predNb]}.', fontsize=16)
        ax1.hist(modelPred[:, predNb])
        ax1.set_title('Neural Net prediction')
        ax2.hist(yDataTest[:, predNb])
        ax2.set_title('Original values')
        plt.savefig(saveDir + f'Hist_PredvOrig_{attrList[predNb]}.pdf')
        plt.show()

    # Plot the inidividual differences between the predictions and values.
    for predNb in range(modelPred.shape[1]):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(f'Divergences for {attrList[predNb]}.', fontsize=16)
        ax1.plot(yDataTest[:, predNb] - modelPred[:, predNb], c='C1')
        ax1.set_title('Individual divergences. y_Orig - y_Pred')

        absDist = np.abs(yDataTest[:, predNb] - modelPred[:, predNb])
        ax2.hist(absDist)
        ax2.set_title('Individual divergences absolute distribution.')
        infoStr = f'Mean: {np.mean(absDist):.3f} \n Std: {np.std(absDist):.3f}'
        ax2.text(0.8, 0.9, infoStr, transform=plt.gca().transAxes)
        plt.savefig(saveDir + f'Diff_PredvOrig_{attrList[predNb]}.pdf')
        plt.show()

    # Plot the histogram for the individual predictions.
    for predNb in range(modelPred.shape[1]):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f'Predictions for {attrList[predNb]}.', fontsize=16)


if __name__ == '__main__':
    # ------------------------------------------ Data: Load, prep, process ------------------------------------------
    # Set the random seed and load the data
    rndSeed = 0
    np.random.seed(rndSeed)

    # Load up the data, and select the label attributes.
    # attrList = ['p_mH1', 'p_mH2', 'p_mH3', 'p_mA', 'p_mHc', 'p_tbeta', 'p_m12sq', 'p_vs']
    attrList = ['p_vs']

    dataDict = getNumpyArrays('model.out', attrList)
    xData, paramsList = dataDict['xData'], dataDict['ParamList']

    # Make the χ² dictionary.
    chiSq_attrDict = setMnStd(dataDict['yData'], attrList)
    yData = dataDict['yData']

    # Random fraction of ~ (1 - testSplit) is allocated to the testing suite.
    testSplit = 0.85
    boolMask = np.array([True if np.random.uniform() < testSplit else False
                         for _ in range(xData.shape[0])])

    trainArray, testArray = xData[boolMask], xData[np.logical_not(boolMask)]
    yDataTrain, yDataTest = yData[boolMask], yData[np.logical_not(boolMask)]

    # Normalise the training and test set (xData)
    dataNormaliser = ArrayNorms(xData)
    trainArray_norm = dataNormaliser.normData(trainArray)
    testArray_norm = dataNormaliser.normData(testArray)
    # trainArray_norm, testArray_norm = trainArray, testArray

    # ------------------------------------------ Net: Initialise and fit  ------------------------------------------
    # Set up fitting parameters
    nbNeur = 200
    actFct = "relu"
    mBatch = 128
    nbEpochs = 5000
    n_dimm = xData.shape[1]
    y_dimm = 1 #  yData.shape[1]
    nbLayers = 3

    # Initialise the architecture.
    fitModel = makeLinArch(n_dimm, nbNeur, nbLayers + 1, y_dimm)

    # Train a the fit; Set verbose=0 to suppress training output

    from halo import Halo
    spinner = Halo(text=f'Training hard for the money so hard for the moneyehhh 💵💵💵', spinner='moon')
    spinner.start()

    earlyStopping = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=75, verbose=1)]
    history = fitModel.fit(trainArray_norm, yDataTrain, batch_size=mBatch, epochs=nbEpochs,
                           validation_split=0.2, verbose=1, callbacks=earlyStopping)

    modelDir = 'FitModel/MultiDim/' + attrList[0] + '/'
    subprocess.call('mkdir ' + modelDir, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    fitModel.save(modelDir)
    spinner.stop_and_persist(symbol='🌞', text='It done gud.')
    showStats(fitModel, history, testArray_norm, yDataTest, attrList, attrList[0])

    # -------->     TO DO    <-------- #
    # 1) Try multiple layers for problematic predictions.
    # 2) Add plot with histogram of predictions for each of the ones.
