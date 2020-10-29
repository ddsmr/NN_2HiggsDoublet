import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rc
import matplotlib as mpl
# mpl.style.use('seaborn')

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def loadData(paramsList):
    '''
        Load the data from the csv file and returns the data frame and parameter numpy array values.
    '''
    dataFrame = pd.read_csv('../TrainingData/model.out', delimiter='\t', index_col=False)
    return dataFrame[paramsList].values, dataFrame


def makePolyList(xArr, yArr):
    '''
        Return tuples of the pairs corresponding to the xArr and yArr.
    '''
    tupleList = []
    for xCoord, yCoord in zip(xArr, yArr):
        tupleList.append((xCoord, yCoord))
    return tupleList


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


class DataBound:
    '''
        Class to handle points inside bounds.
    '''

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

    def _getValsKeys(self, keyA, keyB):
        '''
            Return the numpy array corresponding to the values specified by the keys (i.e. name of the collumns).
        '''
        return self.dataFrame[[keyA, keyB]].values

    def _getHullList(self, npArray2D):
        '''
            Return a numpy array of points corresponding to the convex hull of the points
        '''

        polyList = []
        hull = ConvexHull(npArray2D[:, 0:2])
        for simplex in hull.simplices:
            polyList.append((npArray2D[simplex, 0][0], npArray2D[simplex, 1][0]))
            polyList.append((npArray2D[simplex, 0][1], npArray2D[simplex, 1][1]))

        return np.array(list(set(polyList)))

    def inBounds(self, keyA, keyB, testArray):
        '''
            Checks if the points in the testArray are contained within the 2D complex hull descibred by the projection
            on the keyA and keyB plane.
        '''
        planeArray = self._getValsKeys(keyA, keyB)
        hullList = self._getHullList(planeArray)

        boolArray = in_hull(testArray, hullList)

        # print('******************')
        # print(testArray.shape, boolArray.shape)
        # print('******************')

        passPoints = testArray[boolArray]
        failPoints = testArray[np.logical_not(boolArray)]

        return {'Pass': passPoints, 'Fail': failPoints, 'BoolMask': boolArray}


if __name__ == '__main__':

    paramsList = ['p_mH2', 'p_mH3', 'p_mA', 'p_mHc',  'p_m12sq', 'p_vs', 'p_tbeta']
    paramVals, dataFrame = loadData(paramsList=paramsList)
    testPoints = np.random.uniform(low=(100, 100), high=(1000, 1000), size=(400, 2))

    datBounder = DataBound(dataFrame)
    passFailDict = datBounder.inBounds('p_mH2', 'p_mH3', testPoints)
    passPoints, failPoints = passFailDict['Pass'], passFailDict['Fail']

    dataFrame.plot.scatter(x='p_mH2', y='p_mH3', edgecolors='C0', s=15, marker='+')
    pScatt = plt.scatter(passPoints[:, 0], passPoints[:, 1], c='C1', label='Inside')
    fScatt = plt.scatter(failPoints[:, 0], failPoints[:, 1], c='C2', label='Outside')
    plt.legend()
    plt.show()
    # exit()
    #
    # # polyList = makePolyList(paramVals[:, 0], paramVals[:, 1])
    # # testPoints_Fail = np.array([[600, 600], [700, 700]])
    # # testPoints_Succ = np.array([[400, 900], [375, 960]])
    #
    # hull = ConvexHull(paramVals[:, 0:2])
    # polyList = []
    # for simplex in hull.simplices:
    #     plt.plot(paramVals[simplex, 0], paramVals[simplex, 1], 'k-')
    #     # print(paramVals[simplex, 0])
    #     # print(paramVals[simplex, 1])
    #     polyList.append((paramVals[simplex, 0][0], paramVals[simplex, 1][0]))
    #     polyList.append((paramVals[simplex, 0][1], paramVals[simplex, 1][1]))
    #
    # polyList = np.array(list(set(polyList)))
    # boolArray = in_hull(testPoints, polyList)
    #
    # passPoints = testPoints[boolArray]
    # failPoints = testPoints[np.logical_not(boolArray)]
    #
    # print(passPoints.shape, failPoints.shape)
    #
    # pScatt = plt.scatter(passPoints[:, 0], passPoints[:, 1], c='C1', label='Inside')
    # fScatt = plt.scatter(failPoints[:, 0], failPoints[:, 1], c='C2', label='Outside')
    # plt.legend()
    # plt.show()
    # exit()
