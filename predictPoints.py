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


def loadPoints(filterDict=None):
    '''
        Load the points predicted via the chi square minimisation.
    '''
    with open('Predictions/predPoints.json', 'r') as jsonIn:
        pointDict = json.load(jsonIn)

    if filterDict is not None:
        for filterAttr in filterDict.keys():
            if filterDict[filterAttr]['Min'] is not None:
                hasMin = True
            else:
                hasMin = False
            if filterDict[filterAttr]['Max'] is not None:
                hasMax = True
            else:
                hasMax = False

        for pointID in pointDict.keys():
            for filterAttr in filterDict.keys():
                print(pointDict[pointID][filterAttr], filterDict[filterAttr]['Min'])
                if hasMin and pointDict[pointID][filterAttr] < filterDict[filterAttr]['Min']:
                    del pointDict[pointID]
                if hasMax and pointDict[pointID][filterAttr] > filterDict[filterAttr]['Max']:
                    del pointDict[pointID]

    print(len(pointDict))
    return pointDict['PointDict']


def convDictArray(pointDict):
    '''
        Convert the point dictionary into a numpy array.
    '''
    pointArr = []
    paramList = ['p_L1', 'p_L2', 'p_L3', 'p_L4', 'p_L5', 'p_L6', 'p_L7', 'p_L8', 'p_u11sq', 'p_u22sq', 'p_m12sq',
                 'p_ussq']

    for pointId in pointDict.keys():
        paramVec = []
        for param in paramList:
            paramVec.append(pointDict[pointId][param])
        paramVec = np.array(paramVec)
        pointArr.append(paramVec)
    pointArr = np.array(pointArr)

    return pointArr, paramList.index('p_m12sq')


def exportDictPred(predDict, attrList, nbPoints):
    '''
        Convert dictionary into table and export it as a pred.out file
    '''
    predTable = np.array([[] for _ in range(nbPoints)])

    for wrtAttr in attrList:
        predArr = predDict[wrtAttr]
        predTable = np.concatenate((predTable, predArr), axis=1)

    predTable = np.array(predTable)
    np.savetxt('Predictions/predPoints_wAttrs.txt', predTable, delimiter=' ', header=' '.join(attrList))
    return None


if __name__ == '__main__':
    # Load up all the fits
    attrList = ['p_mH2', 'p_mH3', 'p_mA', 'p_mHc', 'p_tbeta', 'p_vs']  # , 'p_m12sq'

    filterDict = {'p_m12sq': {'Min': 0.0, 'Max': None}}
    pointDict = loadPoints()
    pointArray, m12Pos = convDictArray(pointDict)

    predResDict = {'p_m12sq': pointArray[:, m12Pos].reshape(len(pointDict), 1)}

    # Normalise the training and test set (xData)
    dataNormaliser = ArrayNorms(pointArray)
    pointArray = dataNormaliser.normData(pointArray)

    fitDict = {}
    for attrToFit in attrList:
        fitDict[attrToFit] = keras.models.load_model('FitModel/MultiDim/' + attrToFit)

    for attrToFit in attrList:
        fitRes = fitDict[attrToFit].predict(pointArray)
        predResDict[attrToFit] = fitRes

    attrList.append('p_m12sq')
    exportDictPred(predResDict, attrList, len(pointDict))
