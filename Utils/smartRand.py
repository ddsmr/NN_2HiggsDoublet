from math import *
from pprint import pprint as pp
from colorama import Fore, Style
import sys, os
import time
import numpy as np
import math

import random
from random import randint
from copy import deepcopy


from math import sin, cos
from scipy.special import jv as BesselJ
from scipy.special import yv as BesselY


nbrTerminalRows, nbrTerminalCols = os.popen('stty size', 'r').read().split()
nbrTerminalRows = int(nbrTerminalRows)
nbrTerminalCols = int(nbrTerminalCols)
delimitator = '\n' + '▓' * nbrTerminalCols + '\n'
FNULL = open(os.devnull, 'w')


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def printCentered(stringToPrint, color='' , fillerChar=' '):
    '''
        Utility function that prints with a color the string defined in stringToPrint, centered in the console, with a fillerChar.

        Arguments:
            - stringToPrint     ::           String to be printed in the center of the console line.
            - color             ::           Color in which the string will be printed in.
            - fillerChar        ::           Character to fill the rest of the console line.

        Returns:
            - None

    '''
    rows, columnsTerm = os.popen('stty size', 'r').read().split()

    stringToPrint = fillerChar *  int ( (int(columnsTerm) - len(stringToPrint)) / 2) + color + stringToPrint + Style.RESET_ALL + fillerChar *  int ( (int(columnsTerm) - len(stringToPrint)) / 2)

    print(stringToPrint)

    return None

def _evalExpr(exprToEval, paramList, paramDict):
    '''
        Auxiliary funciton used to evaluate a mathematical expression passed as a string (exprToEval), where the variables are defined in the paramList list, and the values of the parameters are held in paramDict.

        Arguments:
        -   exprToEval              ::              TYPE: str. Expression to be evaluated. E.g. "Mu**2 - sin(Theta)"
        -   paramList               ::              TYPE: list of strings. List of the symbolic name of the variables. E.g. ["Mu", "Theta"].
        -   paramDict               ::              TYPE: dict. Ditionary containing the values of the parameters. E.g. {"Mu":9.3, "Theta":1.2}

        Returns:
        - float

    '''

    for paramToAttr in paramList:
        exec (paramToAttr + '=' + str (paramDict[paramToAttr]) )

    return eval(exprToEval)

class smartRand():
    '''
        Functional class that should be used whenever one wants a random number selection that has intermediate consistency checks.
    '''
    def __init__(self, dictMinMax, condDict, rndDictDecisionTree, toSetDict, tryCountLimit=10):
        '''
            To perform a SmartRandomGenerator we need the following:

            Arguments:
            -   dictMinMax              ::              TYPE: dict. Contains the bounds of the parameters, i.e. min and max.
            -   condDict                ::              TYPE: dict. Contains the intermediate conditions that are to be checked. Has the following general structure:

                    condDict = {'rndCond1': {
                                        'ParamList' : [ "param1", ... ],
                                        'ToCheckExpr' : "f(param1, ...)",
                                        'ToCheckMinBound' : minBound / None,
                                        'ToCheckMax' : maxBound / None,
                                        'Description' : 'User supplied description.'
                                        },
                                ...
                                }
                    which can be extended with case checking:

                    'ParamCaseCheck' : {'ParamList' : ['paramToCheck1', ...],
                                        'Cases' : {'cond1(paramToCheck1, ...)':  'Case-1',
                                                   'cond2(paramToCheck1, ...)' : 'Case-2'}
                                        },
                    'ToCheckExpr' : {"Case-1" : "f1(param1, ...)",
                                     "Case-2" : "f2(param1, ...)" },

            -   rndDictDecisionTree     ::              TYPE: OrderedDict. Contains the actual decision tree. Should have the general structure:

                    rndDictDecisionTree = OrderedDict   ([
                          ('Check-0', {'ToPick': ['k', 'zL'],                   <---- The parameters to pick from dictMinMax range
                                       'ToCheck': ['mKK','sinTh1'],             <---- The conditions to check from condDict
                                       'ToSet' : [],                            <---- Parameters to set via toSetDict. CAN BE EMPTY.
                                       'Pass' : 'Check-1',                      <---- Check to proceed to if we have a succesfull
                                                                                      condition check.
                                       'Fail' : 'Check-0'}                      <---- Check to proceed to if we have a failed
                                                                                      condition check.
                          ),
                          ('Check-1', {'ToPick' : ['c1', 'Mu1'],
                                       'ToSet' : ['Mu11'],
                                       'ToCheck': ['sinThdiv2-3'],
                                       'Pass' : 'Success',                      <---- Algorithm will stop when we he hit 'Success'.
                                       'Fail' : 'Check-1'}
                          ),
                          ...
                    ])

            - toSetDict                 ::              TYPE: dict. Contains the parameters that we want to set against other generated parameters. Has the general structure:

                    toSetDict = {
                        'paramToSet':{
                                        'ParamList' : ['paramToSetAgainst1', 'paramToSetAgainst2', ...],
                                        'ToSetExpr' :  "f(paramToSetAgainst1, paramToSetAgainst2, ...)"

                        }
                    which can be extended with case checking:

                    'ParamCaseCheck' : {'ParamList' : ['paramToCheck1', ...],
                                        'Cases' : {'cond1(paramToCheck1, ...)':  'Case-1',
                                                   'cond2(paramToCheck1, ...)' : 'Case-2'}
                                        },
                    'ToSetExpr' :   {"Case-1" : "f1(param1, ...)",
                                     "Case-2" : "f2(param1, ...)" },
                    }

            - tryCountLimit             ::              TYPE: int. DEFAULT:10. Number of tries before the algorithm goes to the fail branch.
        '''

        self.dictMinMax = dictMinMax
        self.condDict = condDict
        self.rndDictDecisionTree = rndDictDecisionTree
        self.toSetDict = toSetDict
        self.tryCountLimit = tryCountLimit

        # return None

    def genRnd(self,  param,  wrkingPrec = 4, samplingPDF = 'Uniform'):
        '''
            Generates a random value for the param parameter within self.dictMinMax with it's specified bounds, at a precision of wrkingPrec.

            Arguments:
            -   param           ::              Type: str. Parameter within self.dictMinMax
            -   wrkingPrec      ::              Type: int. DEFAULT: 4. Working precision for the rounding.

            Returns:
            -  float

        '''

        dictMinMax = self.dictMinMax

        if samplingPDF == 'Uniform':
            return round ( random.uniform(dictMinMax[param]['Min'], dictMinMax[param]['Max'])  , wrkingPrec)
        elif samplingPDF == 'Normal':
            sigmaPDF = ( dictMinMax[param]['Max'] - dictMinMax[param]['Min']) / 2
            muPDF = dictMinMax[param]['Min'] + sigmaPDF

            # s = np.random.normal(muPDF, sigmaPDF, 10000)
            # import matplotlib.pyplot as plt
            # count, bins, ignored = plt.hist(s, 30, density=True)
            # plt.plot(bins, 1/(sigmaPDF * np.sqrt(2 * np.pi)) *np.exp( - (bins - muPDF)**2 / (2 * sigmaPDF**2) ),
            #          linewidth=2, color='r')
            # plt.show()
            return round( np.random.normal(muPDF, sigmaPDF), wrkingPrec)

    def genRandUniform_Sn(self, paramList):
        '''
            Generates randomly distributed points on the surface of an n sphere , i.e. on S^(n-1). Requires a parameter list to associate with the random directions on S^(n-1). Based on Marsaglia's algorithm.

            Arguments:
            -   paramList           ::          Type: list. List of parameters to pick a random direction in the unit hypershpere R^n.

            Returns:
            -   randDict            ::          Type: dict. Dictionary with the random directions in the unit R^n.
        '''


        randDict = {}
        for param in paramList:
            randDict[param] = []

        for rndParam in randDict.keys():
            randDict[rndParam] = np.random.standard_normal()

        vecMagSquare = 0
        for rndParam in randDict.keys():
            vecMagSquare += (randDict[rndParam])**2

        for rndParam in randDict.keys():
            randDict[rndParam] = randDict[rndParam] / math.sqrt( vecMagSquare )

        return randDict

    def genRandUniform_Rn(self, paramDict, rSigma, noGoZone_Sn = []):
        '''
            Uses Marsaglia's algorihm to generate a random deviation from the seed point paramDict with deviations in  R^n space defined via the paramDict. The 'deviation length' is sampled from a normal distribution with 0 mean and rSigma deviation.

            Arguments:
            -   paramDict           ::          Type: dict. Param dictionary that contains the seed param values.
            -   rSigma              ::          Type: float. Sigma paramter in the deviation length distribution.

            Returns:
            -   paramDictNew        ::          Type: dict. New parameter dictionary with the values generated from the seed point via Marsaglia's and the projection on parameter space.
        '''
        while True:

            randDictDirections = self.genRandUniform_Sn( list(paramDict.keys()) )

            if True:###  Put here exclusion condition
                break
            else:
                continue

        # print ('Working with an ' + Fore.RED + 'σ_r' + Style.RESET_ALL + ' value of :', rSigma)
        rVal = np.random.normal( 0, rSigma )

        devDict = {}
        paramDictNew = deepcopy(paramDict)

        for param in paramDict.keys():
            devDict[param] =  rVal* randDictDirections[param] * paramDict[param]

        for param in paramDict.keys():
            paramDictNew[param] = paramDict[param] + devDict[param]

        return paramDictNew

    def _checkParamDict(self, paramDict):
        '''
            Checks if the values specified in paramDict pass the conditions specified in the rndDictDecisionTree in th econfig file.

            Arguments:
            -   paramDict               ::              Type: dict. Param dictionary with values to be checked.

            Returns:
            -   passConds               ::              Type: bool. True/False depending if the point obeys the conditions.
        '''
        rndDict = self.rndDictDecisionTree
        checkNb = 'Check-0'


        while checkNb != 'Success':

            passConds = True

            for condToCheck in rndDict[checkNb]['ToCheck']:
                # print (checkNb)

                try:
                    valueToCheck = self.evaluateCondition(condToCheck, paramDict)
                except Exception as e:
                    # print (Fore.GREEN+'%%%%%%%%%%%%%%%'+ Style.RESET_ALL, e)
                    return False

                try:
                    # print (valueToCheck)
                    passConds = passConds and self.checkBounds(condToCheck, valueToCheck)

                except Exception as e:
                    # print (Fore.RED+'%%%%%%%%%%%%%%%' + Style.RESET_ALL, e)
                    return False
            if passConds == True:

                checkNb = rndDict[checkNb]['Pass']
            else:
                return False



        return passConds

    def findBest_rSigma(self,  paramDictCentral, numberOfTests = 1000, passFrac = 0.999936, reductionFactor = 1.688, rSigma = 3.0):
        '''
            Function to find the appropriate rSigma value for a point's set of parameter values specified in paramDictCentral. Function performs a numberOfTests tests (in this case via _checkParamDict to see if the new points are able to produce in principle a new valid point) and checks if the fraction of positive tests is above or under passFrac. If the number of positives is above the the funciton returns the value of rSigma, and if not it repeats the algorihm with a new value of rSigma / reductionFactor untill we get a pass.

            Arguments:
            -   paramDictCentral            ::          Type: dict. Parameter dictionary for a valid point in phase space.
            -   numberOfTests               ::          DEFAULT: 1000. Type: int. Number of total test statistics.
            -   passFrac                    ::          DEFAULT: 0.997. Type: float. Lower bound on the fraction of positive test before we break.
            -   reductionFactor             ::          DEFAULT: 1.688. Type: float. Factor to scale rSigma down with.
            -   rSigma                      ::          DEFAULT: 3.0. Type: float. Initial rSigma value to be reduced.

            Returns:
            -   rSigma                      ::          Type: float. Appropriate rSigma value.
        '''
        # print (Fore.GREEN + 'Original value has ' , self._checkParamDict(paramDictCentral) , ' for its check.' + Style.RESET_ALL)

        # Should probably look at what happens if rSigma is to small to begin with.

        while True:
            passNb = 0

            numberOfTestsToGo = numberOfTests

            while numberOfTestsToGo > 0:
                paramDict = self.genRandUniform_Rn( paramDictCentral , rSigma)

                # print(  Fore.YELLOW + 'Old Central.' + Style.RESET_ALL)
                # pp(paramDictCentral)
                # print(  Fore.YELLOW + 'New dict.' + Style.RESET_ALL)
                #
                # pp(paramDict)
                # print (delimitator)
                # print (self._checkParamDict(paramDict))
                # print ('*')

                if self._checkParamDict(paramDict):
                    passNb += 1
                numberOfTestsToGo += (-1)

            # print('Pass Fraction of: ', (passNb / numberOfTests))

            if (passNb / numberOfTests) < passFrac:
                # time.sleep(1)
                rSigma = rSigma / reductionFactor
                # print (Fore.RED + 'New rSigma: ' + Style.RESET_ALL, rSigma)
            else:
                return rSigma

    def evaluateCondition(self, condToCheck , paramDict):
        '''
            Evaluates the condition from self.condDict specified via the condToCheck, with the values specified in paramDict. If the condition has 'ParamCaseCheck' in its structure, then the function evaluates the corresponding case condition.

            Arguments:
            -   condToCheck             ::          Type: str. Condition that has to be checked. Should be a key in self.condDict.
            -   paramDict               ::          Type: dict. Parameter value dicitonary to be used to evaluate condToCheck.

            Returns:
            -   float                   ::          Returns the value of the evaluated expression corresponding to condToCheck.
        '''
        condDict = self.condDict

        if 'ParamCaseCheck' in  list(condDict[condToCheck].keys()):

            for caseNb in list (condDict[condToCheck]['ParamCaseCheck']['Cases'].keys() ):
                if _evalExpr(  caseNb, condDict[condToCheck]['ParamCaseCheck']['ParamList'], paramDict  ) == True:
                    # print (caseNb)
                    caseStr = condDict[condToCheck]['ParamCaseCheck']['Cases'][caseNb]
                    break


            return _evalExpr(condDict[condToCheck]['ToCheckExpr'][caseStr], condDict[condToCheck]['ParamList'], paramDict)


        else:
            return _evalExpr(condDict[condToCheck]['ToCheckExpr'], condDict[condToCheck]['ParamList'], paramDict)

    def checkBounds(self, condToCheck, valueToCheck):
        '''
            Checks valueToCheck of condToCheck against the bounds specified in self.condDict.

            Arguments:
            -   condToCheck             ::          Type: str. Condition that has to be checked. Should be a key in self.condDict.
            -   valueToCheck            ::          Type: float. Value to be checked against the bounds specified in self.condDict at the condToCheck condition

            Returns:
            -   Bool                    ::          Returns wheater value is within the bounds (consider a pass if the bound is None).
        '''
        condDict = self.condDict


        passBound = True
        for minMaxBound in ['ToCheckMinBound', 'ToCheckMax']:

            if condDict[condToCheck][minMaxBound] == None:
                pass
            else:
                if minMaxBound == 'ToCheckMinBound':
                    if condDict[condToCheck]['ToCheckMinBound'] < valueToCheck:
                        pass
                    else:
                        passBound = False
                elif minMaxBound == 'ToCheckMax':
                    if condDict[condToCheck]['ToCheckMax'] > valueToCheck:
                        pass
                    else:
                        passBound = False

        return passBound

    def setFromRnd(self, paramDict, paramToSet ):
        '''
            Analogous to evaluateCondition. Given a paramToSet the function will set it's value according to the rules specified in self.toSetDict with the values specified in paramDict. If the parameter has 'ParamCaseCheck' in its self.toSetDict structure, then the function evaluates the corresponding case condition.

            Arguments:
            -   paramDict               ::          Type: dict. Parameter value dicitonary to be used to set paramToSet.
            -   paramToSet              ::          Type: str. Parameter key that specifies which parameter is to be set via the rules speicfied in self.toSetDict.

            Returns:
            -   float                   ::          Returns the value of the evaluated expression corresponding to condToCheck.
        '''

        toSetDict = self.toSetDict

        if 'ParamCaseCheck' in  list(toSetDict[paramToSet].keys()):

            for caseNb in list (toSetDict[paramToSet]['ParamCaseCheck']['Cases'].keys() ):

                if _evalExpr(  caseNb, toSetDict[paramToSet]['ParamCaseCheck']['ParamList'], paramDict  ) == True:
                    # print (caseNb)
                    caseStr = toSetDict[paramToSet]['ParamCaseCheck']['Cases'][caseNb]
                    break

            # print ('SETTTTTTT')
            # pp (paramDict)
            return _evalExpr(toSetDict[paramToSet]['ToSetExpr'][caseStr], toSetDict[paramToSet]['ParamList'], paramDict)

        else:
            return  _evalExpr(toSetDict[paramToSet]['ToSetExpr'], toSetDict[paramToSet]['ParamList'], paramDict)

    def genSmartRnd(self, debug=False, samplingPDF = 'Uniform'):
        '''
            Main function that goes through the self.rndDictDecisionTree dicitonary and performs the specified parameter setting and condition checking. Stops when it has found a consistent set of parameters and the algorithm reaches the 'Success' keyword.

            Arguments:
            -   debug               ::              TYPE: Bool. DEFAULT: False. Set to True to enable debugging printing.

            Returns:
            -   dict                ::              Returns the paramDict with the generated consistent set of parameters.
        '''



        rndDict = self.rndDictDecisionTree


        startTime = time.time()
        tryCountLimit = self.tryCountLimit
        checkNb = 'Check-0'

        paramDict = {}
        reinitCount = True
        # print (delimitator)
        nbOfTotalChecks = 0

        while checkNb != 'Success':
            # printCentered (checkNb, Fore.BLUE, '-')

            if reinitCount == True:
                tryCount = 1

            if rndDict[checkNb]['ToPick']:
                for paramToGen in  rndDict[checkNb]['ToPick']:
                    paramDict[paramToGen] = self.genRnd(paramToGen, samplingPDF = samplingPDF)

            if rndDict[checkNb]['ToSet']:
                for paramToSet in rndDict[checkNb]['ToSet']:
                    try:
                        paramVal = self.setFromRnd(paramDict, paramToSet)
                        # print (paramVal)
                        paramDict[paramToSet] = paramVal
                    except Exception as e:
                        print(e)
                        checkNb = 'Check-0'
                        nbOfTotalChecks += 1



            # pp (paramDict)

            passConds = True
            for condToCheck in rndDict[checkNb]['ToCheck']:
                # printCentered('=====>   Checking :'+ condToCheck + '   <=====')


                try:
                    valueToCheck = self.evaluateCondition(condToCheck, paramDict)
                except Exception as e:
                    print(e)
                    reinitCount = True
                    checkNb = 'Check-0'
                    nbOfTotalChecks += 1
                    break
                # print (valueToCheck, condToCheck)

                try:
                    passConds = passConds and self.checkBounds(condToCheck, valueToCheck)
                except Exception as e:
                    print (e)
                    checkNb = 'Check-0'
                    nbOfTotalChecks += 1
                    passConds = False
                    break



            # print (Fore.YELLOW + 'Performed {0} checks so far'.format(tryCount) + Style.RESET_ALL)
            if passConds == True:
                checkNb = rndDict[checkNb]['Pass']
                reinitCount = True
                nbOfTotalChecks += 1

                if debug == True:

                    print (Fore.GREEN + '✔✔✔✔✔✔✔✔' + Style.RESET_ALL, 'Moving on to ', checkNb)
                    print (delimitator)


            elif tryCount < tryCountLimit :
                pass
                tryCount += 1
                reinitCount = False
                nbOfTotalChecks += 1

                if debug == True:
                    print (Fore.YELLOW + '▤▤▤▤▤▤▤▤' + Style.RESET_ALL, 'Trying again ', checkNb)
            else:
                reinitCount = True
                nbOfTotalChecks += 1
                checkNb = rndDict[checkNb]['Fail']

                if debug == True:
                    print (Fore.RED + '🚫🚫🚫🚫' + Style.RESET_ALL, 'Stepping back to ', checkNb)


        # pp (paramDict)
        if debug == True:
            printCentered( '🎵🎵🎵   Total number of  ' + str(nbOfTotalChecks) + '  checks   🎵🎵🎵')
            pp (paramDict)

        return paramDict
