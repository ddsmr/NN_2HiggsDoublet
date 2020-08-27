import numpy as np
from time import gmtime, strftime

# Pretty python traceback outputs
try:
    import colored_traceback
    colored_traceback.add_hook()
except Exception as e:
    print(e)


class ArrayNormaliser:
    '''
        Normalisation class, has the direct and the inverse transformations included as methods.
    '''

    def __init__(self, datArray, normType):
        '''
            Initialise the normalisation class with the array values and the normalisation type. Save the required
            attributes in the normalisation dictionary.

            If normType is:
                - 'SetMeanStd': normalise data via Z = (X - μ) / σ.
                - 'UnitVec': normalise each matrix as M / det(M).
        '''
        self.normType = normType
        self.normDict = {}
        self.setID = ''
        if normType == 'SetMeanStd':
            # Save the SET's mean and standard deviation
            self.normDict['Norm'] = {'Mean': np.ndarray.mean(datArray, axis=0),
                                     'Std': np.std(datArray)}
        elif normType == 'MeanStdDimms':
            # Save the individual dimmensions mean and standard deviation
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
            Normalise the data via the direct transformation.
        '''
        if self.normType == '':
            return {'NormArray': datArray, 'SetID': ''}
        elif self.normType == 'SetMeanStd':
            normArray = (datArray - self.normDict['Norm']['Mean']) / self.normDict['Norm']['Std']
            return {'NormArray': normArray, 'SetID': ''}

        elif self.normType == 'UnitVec':
            currTime = strftime("-%d%m%Y%H%M%S", gmtime())
            self.setID = 'DataSet-' + str(datArray.shape) + '-' + str(int(np.random.uniform(1, 1000))) + currTime
            nbMatrices = datArray.shape[0]
            normArray = []

            for matNb in range(nbMatrices):
                unitNorm = np.linalg.det(datArray[matNb])
                self.normDict[str(matNb)] = unitNorm
                normArray.append(datArray[matNb] / unitNorm)

            return {'NormArray': np.array(normArray), 'SetID': self.setID}

        elif self.normType == 'MeanStdDimms':
            normArray = []

            if len(datArray.shape) == 2:
                nbAttrs = datArray.shape[1]

                for attNb in range(nbAttrs):
                    attMean = self.normDict[str(attNb)]['Mean']
                    attStd = self.normDict[str(attNb)]['Std']

                    normAttArr = (datArray[:, attNb] - attMean) / attStd
                    normArray.append(normAttArr)

                normArray = np.transpose(np.array(normArray))

            elif len(datArray.shape) == 1:
                attMean = self.normDict['0']['Mean']
                attStd = self.normDict['0']['Std']
                normAttArr = (datArray - attMean) / attStd
                normArray = np.array(normAttArr)

            return normArray

    def invNormData(self, invDatArray, setID):
        '''
            Perform the inverse normalisation on the inverse data array.
        '''
        if self.normType == '':
            return invDatArray
        elif self.normType == 'SetMeanStd':
            deNormArray = invDatArray * self.normDict['Norm']['Std'] + self.normDict['Norm']['Mean']
            return deNormArray
        elif self.normType == 'UnitVec':
            if setID != self.setID:
                raise KeyError('The inverse data set does not correspond to the original set.')
            else:
                nbMatrices = invDatArray.shape[0]
                deNormArray = []
                for matNb in range(nbMatrices):
                    unitNorm = self.normDict[str(matNb)]
                    deNormArray.append(invDatArray[matNb] * unitNorm)
                print(len(deNormArray))
                return np.array(deNormArray)

        elif self.normType == 'MeanStdDimms':
            transDict = self.normDict
            if transDict is None:
                print('Class instance does not have a direct transformation dictionary. Run normData() on the original set.')
                raise

            if len(invDatArray.shape) == 2:
                nbAttrs = invDatArray.shape[1]
                normArray = []

                for attrNb in range(nbAttrs):
                    attrMean, attrStd = transDict[str(attrNb)]['Mean'], transDict[str(attrNb)]['Std']
                    normAttArr = invDatArray[:, attrNb] * attrStd + attrMean
                    normArray.append(normAttArr)
                normArray = np.transpose(np.array(normArray))

            elif len(invDatArray.shape) == 1:
                attrMean, attrStd = transDict['0']['Mean'], transDict['0']['Std']
                normArray = invDatArray[:, ] * attrStd + attrMean

            return normArray


if __name__ == '__main__':
    '''
       Test the functionality of the normalisation class.
    '''
    xArrTest = np.random.uniform(size=(100, 5, 5))
    matNormer = ArrayNormaliser(xArrTest, 'UnitVec')
    normDict = matNormer.normData(xArrTest)

    denormData = matNormer.invNormData(normDict['NormArray'], normDict['SetID'])
    print(xArrTest[0], '\n\n', normDict['NormArray'][0], '\n\n', denormData[0])
