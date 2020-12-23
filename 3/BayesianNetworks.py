from functools import reduce

import numpy as np
import pandas as pd


# Function to create a conditional probability table
# Conditional probability is of the form p(x1 | x2, ..., xk)
# varnames: vector of variable names (strings) first variable listed
#           will be x_i, remainder will be parents of x_i, p1, ..., pk
# probs: vector of probabilities for the flattened probability table
# outcomesList: a list containing a vector of outcomes for each variable
# factorTable is in the type of pandas dataframe
# See the example file for examples of how this function works


def adjustDataFrameColumn(df,name,index):
    col = list(df[name])
    del df[name]
    df.insert(index,name,col)

def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs
    
    adjustDataFrameColumn(factorTable,'probs',0)
    adjustDataFrameColumn(factorTable,factorTable.columns[-1],0)

    return factorTable


# Build a factorTable from a data frame using frequencies
# from a data frame of data to generate the probabilities.
# data: data frame read using pandas read_csv
# varnames: specify what variables you want to read from the table
# factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i, 'probs'] = sum(a == (i + 1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j, 'probs'] = factorTable['probs'][j] / normalizeZ

    adjustDataFrameColumn(factorTable,factorTable.columns[-1],0)
    return factorTable


def joinProbxProby(df):
    probx = df['probs_x']
    proby = df['probs_y']
    probs = [0 for i in range(len(probx))]

    for i in range(len(probx)):
        probs[i] = probx[i] * proby[i]

    df['probs'] = probs
    del df['probs_x']
    del df['probs_y']

    return df


def reoriganize(df,variable):
    newdf = pd.DataFrame({})
    cond = []
    for name in df.columns:
        if name in variable:
            newdf[name] = df[name]
        elif name != 'probs':
            cond.append(name)
    newdf['probs'] = df['probs']
    for name in cond:
        newdf[name] = df[name]
    return newdf



def getVariableAndCondition(df):
    labels = list(df.columns)
    probsIndex = labels.index('probs')
    return [labels[:probsIndex],labels[probsIndex + 1:]]

# Join of two factors
# factor1, factor2: two factor tables
#
# Should return a factor table that is the join of factor 1 and 2.
# You can assume that the join of two factors is a valid operation.
# Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    f1 = pd.DataFrame.copy(factor1)
    f2 = pd.DataFrame.copy(factor2)

    joinFactor = None
    # TODO: start your code
    joinProb,condProb = None,None
    hasJoin,hasCond = False,False

    if f1.columns[-1] == 'probs':
        joinProb = f1
        hasJoin = True
    else:
        condProb = f1
        hasCond = True

    if f2.columns[-1] == 'probs':
        joinProb = f2
        hasJoin = True
    else:
        condProb = f2
        hasCond = True

    if not hasCond:
        f1['helper'] = 1
        f2['helper'] = 1
        joinFactor = pd.merge(f1,f2,how = 'left',on = 'helper')
        del joinFactor['helper']
        joinProbxProby(joinFactor)
        return joinFactor
    if not hasJoin:
        return None

    if1 = list(joinProb.columns)
    if1.remove('probs')
    [variable,cond2] = getVariableAndCondition(condProb)
    joinTarget = []
    for label1 in if1:
        if label1 in cond2:
            joinTarget.append(label1)
        if label1 in variable:
            condProb = marginalizeFactor(condProb,label1)
            variable.remove(label1)

    joinFactor = joinProb.merge(condProb,on = joinTarget)
    joinProbxProby(joinFactor)
    if1.extend(variable)
    # end of your code
    joinFactor = reoriganize(joinFactor,if1)
    return joinFactor


# Marginalize a variable from a factor
# table: a factor table in dataframe
# hiddenVar: a string of the hidden variable name to be marginalized
#
# Should return a factor table that marginalizes margVar out of it.
# Assume that hiddenVar is on the left side of the conditional.
# Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    factor = pd.DataFrame.copy(factorTable)
    if hiddenVar not in list(factor.columns):
        return factor

    # TODO: start your code
    labels = list(factor.columns)
    labels.remove(hiddenVar)
    labels.remove('probs')

    if len(labels) == 0:
        return pd.DataFrame()

    factor['probs'] = factor.groupby(labels)['probs'].transform('sum')
    del factor[hiddenVar]
    factor.drop_duplicates(subset = labels,keep = 'first',inplace = True)

    # end of your code
    return factor


def getJoinTableOfVariable(bayesNet,var,conditions,variables):
    index = len(variables)
    for i in range(len(variables)):
        if var in variables[i] and len(conditions[i]) == 0:
            index = i
            break
    if index == len(variables):
        for i in range(len(variables)):
            if var in variables[i]:
                index = i
                break
    if index == len(variables):
        return (None,-1)

    table = bayesNet[index]
    if table.columns[-1] == 'probs':
        return (pd.DataFrame.copy(table),index)
    tableConditions = conditions[index]
    for cond in tableConditions:
        (condTable,condIndex) = getJoinTableOfVariable(bayesNet,cond,conditions,variables)
        if condIndex < 0:
            return (None,-1)
        table =  joinFactors(table,condTable)
    return (table,index)
        
def delVariablesFromNet(Net,conditions,variables,targets):
    targetIndices = []
    for label in targets:
        for i in range(len(variables)):
            if label in variables[i] and i not in targetIndices:
                targetIndices.append(i)
                break

    targetIndices = sorted(targetIndices)
    for i in range(len(targetIndices) - 1,-1,-1):
        del Net[targetIndices[i]]
        del variables[targetIndices[i]]
        del conditions[targetIndices[i]]

# Marginalize a list of variables
# bayesnet: a list of factor tables and each table in dataframe type
# hiddenVar: a string of the variable name to be marginalized
#
# Should return a Bayesian network containing a list of factor tables that results
# when the list of variables in hiddenVar is marginalized  of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    if isinstance(hiddenVar, str):
        hiddenVar = [hiddenVar]

    if not bayesNet or not hiddenVar:
        return bayesNet

    marginalizeBayesNet = bayesNet.copy()

    # TODO: start your code
    for var in hiddenVar:
        variables = []
        conditions = []
        for table in marginalizeBayesNet:
            [variable,condition] = getVariableAndCondition(table)
            variables.append(variable)
            conditions.append(condition)

        (varTable,index) = getJoinTableOfVariable(marginalizeBayesNet,var,conditions,variables)
        if index < 0:
            return bayesNet

        delVariables = list(varTable.columns)
        delVariables.remove('probs')
        #for label in variables[index]:
        #    delVariables.remove(label)
        #if len(variables[index]) == 1:
        #    delVariables.append(label)
        #else:
        #    del marginalizeBayesNet[index][var]
        delVariablesFromNet(marginalizeBayesNet,conditions,variables,delVariables)
        for i in range(len(conditions)):
            targetTable = marginalizeBayesNet[i]
            if var in conditions[i]:
                targetTable = marginalizeFactor(joinFactors(varTable,targetTable),var)
                [variables[i],conditions[i]] = getVariableAndCondition(targetTable)
            for variable in varTable.columns[:-1]:
                if variable in variables[i]:
                    targetTable = marginalizeFactor(targetTable,variable)
                    [variables[i],conditions[i]] = getVariableAndCondition(targetTable)
            marginalizeBayesNet[i] = targetTable
        for i in range(len(marginalizeBayesNet) - 1,-1,-1):
            if marginalizeBayesNet[i].empty:
                del marginalizeBayesNet[i]
        varTable = marginalizeFactor(varTable,var)
        if not varTable.empty:
            marginalizeBayesNet.append(varTable)
    # end of your code

    return marginalizeBayesNet


# Update BayesNet for a set of evidence variables
# bayesNet: a list of factor and factor tables in dataframe format
# evidenceVars: a vector of variable names in the evidence list
# evidenceVals: a vector of values for corresponding variables (in the same order)
#
# Set the values of the evidence variables. Other values for the variables
# should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):
    if isinstance(evidenceVars, str):
        evidenceVars = [evidenceVars]
    if isinstance(evidenceVals, str):
        evidenceVals = [evidenceVals]

    updatedBayesNet = bayesNet.copy()
    # TODO: start your code
    for i in range(len(evidenceVars)):
        eVar,eVal = evidenceVars[i],evidenceVals[i]
        variables = []
        conditions = []
        for table in updatedBayesNet:
            [variable,condition] = getVariableAndCondition(table)
            variables.append(variable)
            conditions.append(condition)

        (eTable,eIndex) = getJoinTableOfVariable(updatedBayesNet,eVar,conditions,variables)

        delVariables = list(eTable.columns)
        delVariables.remove('probs')
        delVariablesFromNet(updatedBayesNet,conditions,variables,delVariables)


        assignedList = []
        eVarVals = list(eTable[eVar])
        for i in range(len(eVarVals)):
            if eVarVals[i] == int(eVal):
                assignedList.append(i)
        eTable = eTable.iloc[assignedList]
        updatedBayesNet.append(eTable)
        
        #delVars = list(eTable.columns)
        #delVars.remove('probs')
        #for label in variables[eIndex]:
        #    delVars.remove(label)
        #delVariablesFromNet(updatedBayesNet,conditions,variables,delVars)

        updatedBayesNet = marginalizeNetworkVariables(updatedBayesNet,eVar)
    # end of your code

    return updatedBayesNet


# Run inference on a Bayesian network
# bayesNet: a list of factor tables and each table iin dataframe type
# hiddenVar: a string of the variable name to be marginalized
# evidenceVars: a vector of variable names in the evidence list
# evidenceVals: a vector of values for corresponding variables (in the same order)
#
# This function should run variable elimination algorithm by using
# join and marginalization of the sets of variables.
# The order of the elimiation can follow hiddenVar ordering
# It should return a single joint probability table. The
# variables that are hidden should not appear in the table. The variables
# that are evidence variable should appear in the table, but only with the single
# evidence value. The variables that are not marginalized or evidence should
# appear in the table with all of their possible values. The probabilities
# should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    if not bayesNet:
        return bayesNet

    inferenceNet = bayesNet.copy()
    factor = None
    # TODO: start your code
    inferenceNet = evidenceUpdateNet(inferenceNet,evidenceVars,evidenceVals)
    inferenceNet = marginalizeNetworkVariables(inferenceNet,hiddenVar)
    
    factor = inferenceNet[0]

    probs = list(factor[factor.columns[-1]])
    sum = 0
    for i in range(len(probs)):
        sum += probs[i]
    for i in range(len(probs)):
        probs[i] /= sum

    factor[factor.columns[-1]] = probs
    # end of your code

    return factor
