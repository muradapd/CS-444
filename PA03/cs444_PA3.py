import random
import copy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcdefaults()

# rc('text', usetex=True)  # allow latex commands in pyplot

'''
This simple class represents CPT for binary variables only (true/false)
This file has been edited and completed by Patrick Muradaz
'''


class CPT:
    def __init__(self, varName, priorVars, probTable):
        # 2 raised to the priorVars power should be the size of the provided
        # probTable, where each entry has priorVar entries in the key.

        if 2 ** len(priorVars) != len(probTable):
            raise Exception(('CPT constructor, length of probTable was {0:d} and ' +
                             'expected {1:d}.').format(len(probTable), 2 ** len(priorVars)))

        # each tuple in probTable should have len(priorVars) in it.
        # check the first one
        first_key = list(probTable.keys())[0]
        if len(first_key) != len(priorVars):
            raise Exception(('CPT constructor, length of priorVars ({0:d}) and ' +
                             'probTable keys {1:d} are in error.')
                            .format(len(priorVars), len(first_key)))

        self.varName = varName
        self.priorVars = priorVars
        self.probTable = probTable

    def getProb(self, observed):
        observation = []
        for item in observed:
            observation.append(observed.get(item))
        key = tuple(observation)
        return self.probTable[key]

    """
    method: directSample
    parameters: 
      observed: a dictionary that MUST contain
                a key for each conditioned variable
                in this CPT.  The value associated with each
                key must be True or False.

      Construct a tuple of True and False elements, look up the
      probability in the dictionary probTable.  Sample
      a random value, and using the probability retrieved, 
      determine if this variable (self) should be true or false.
      Make a copy of the dictionary of observed values and 
      append this sampled value into the copy of the dictionary. 
    """

    # accept observed and return a dictionary
    # with observed plus this variable inside
    def directSample(self, observed):
        if len(observed) < len(self.priorVars):
            raise Exception(('direct sample: length of observed vars ({0:d}) must' +
                             ' match priorVars({1:d}).')
                            .format(len(observed), len(self.priorVars)))

        # STUDENT CODE HERE
        prob = self.getProb(observed)
        sample = random.uniform(0, 1)
        observe = False

        if sample < prob:
            observe = True

        newDict = observed.copy()

        # STUDENT CODE HERE: and your sampled value to newDict
        newDict.update({self.varName: observe})

        return newDict


# generate a plot of the estimate and save to filename
def varList(vars):
    var_string = r'$'
    first_var = True
    for v in vars.keys():
        if first_var:
            first_var = False
        else:
            var_string += r', '
        if vars[v]:
            var_string += r'\neg '
        var_string += v
    var_string += r'$'

    return var_string


def plotRunningEstimate(runningEstimate, fileName, varSettings, evidence=None):
    plt.clf()

    x_pos = np.arange(len(runningEstimate))

    # the corresponding height for each item in xObjects
    # varString = r'$'
    # firstVar = True
    # title = r'Probability Estimate for '
    # title += varList(varSettings)

    # if evidence is not None:
        # title += r' $\vert$ ' + varList(evidence)

    plt.plot(x_pos, runningEstimate)
    plt.xlabel('iteration(count)', fontsize=16)

    plt.ylabel('P(X)', fontsize=16)
    plt.title('No Latex')

    plt.savefig(fileName, dpi=400, bbox_inches='tight', pad_inches=0.05)


"""
directSampling
compute direct probability (no evidence) .
Code similar to Prior-Sample in textbook (Figure 14.13) except
perform nbrOfSamples times and count.

  parameters:
  -- orderedVars (topological ordering used to sample)
  -- eventToCalc (variables and assignments)
  -- nbrOfSamples -- number of samples to attempt to generate

 returns:
   -- runningEstimate (python list)

Basic idea, try the following nbrOfSamples times:
  -- generate a sample as follows.  For each variable in the
     topological order, sample a value, using the variables
     generated so far if required).

  
"""


def priorSample(orderedVars):
    # STUDENT CODE HERE
    observed = {}
    new_observed = {}
    final = {}

    for var in orderedVars:
        if var.varName == 'B' or var.varName == 'M':
            new_observed.update(var.directSample(observed))
        elif var.varName == 'I':
            new_observed.update(var.directSample(new_observed))
        elif var.varName == 'G':
            new_obs = var.directSample(new_observed)
            final.update({'G': new_obs.get('G')})
            new_observed.update(new_obs)
        elif var.varName == 'J':
            new_observed.update(var.directSample(final))
    return new_observed


def directSampling(orderedVars, eventToCalc, nbrOfSamples):
    count = 0
    running_estimate = []

    for n in range(nbrOfSamples):
        this_sample = priorSample(orderedVars)

        if this_sample == eventToCalc:
            count += 1
        running_estimate.append(count / (n + 1))

    return running_estimate


"""
compute the condition probability of a variable.
This is performed by generating samples that agree with
the evidence and counting

  parameters:
  -- orderedVars (topological ordering used to sample)
  -- X -- variable(s) to query (along with their value) 
     X is a python dictionary (see PA assignment)
  -- evidenceVars (variables and their values given in query)
     This is a python dictionary
  -- nbrOfSamples -- number of samples to attempt to generate

 returns a tuple:
   -- number of rejections
   -- runningEstimate (python list)

Basic idea, try the following nbrOfSamples times:
  -- generate a sample using a call to something similar 
     to priorSample (you may have written a method like 
     this for directSampling), call this value ps
  -- if ps matches the evidence (else increment rejectionCount)
     -- increment npsE (number of samples matching evidence)
     -- if the query variable X is true, increment npsXE
        (number of samples matching evidence and query (X))

  -- return (rejectionCountPercentage,runningEstimate) 
"""


def rejectSampling(orderedVars, X, evidenceVars, nbrOfSamples, alpha):
    rejection_count = 0
    npsXE = 0
    npsE = 0
    sample_set = {}
    running_estimate = []

    for i in range(nbrOfSamples):
        # STUDENT CODE here
        this_sample = priorSample(orderedVars)
        for key in evidenceVars.keys():
            sample_set.update({key: this_sample.get(key)})
        if sample_set == evidenceVars:
            npsE += 1
            for x in X.keys():
                if this_sample.get(x):
                    npsXE += 1
        else:
            rejection_count += 1

        running_estimate.append((npsXE / (i + 1)) * alpha)
    return rejection_count / nbrOfSamples, running_estimate


def main():

    sunny = CPT('S', (), {(): 0.7})
    job_raise = CPT('R', (), {(): 0.01})
    happy = CPT('H', ('S', 'R'), {(True, True): 1.0, (True, False): 0.7,
                                  (False, True): 0.9, (False, False): 0.1})
    orderedVars = [sunny, job_raise, happy]
    eventToCalc = {'H': True}
    
    # construct the Bayes Net in Figure 14.23 in Russell/Norvig
    brokeElectionLaw = CPT('B', (), {(): 0.9})
    politcallyMotivatedProsecutor = CPT('M', (), {(): 0.1})
    indicted = CPT('I', ('B', 'M'), {(True, True): 0.9, (True, False): 0.5,
                                     (False, True): 0.5, (False, False): 0.1})
    foundGuilty = CPT('G', ('B', 'M', 'I'), {(True, True, True): 0.9,
                                             (True, True, False): 0.0,
                                             (True, False, True): 0.8,
                                             (True, False, False): 0.0,
                                             (False, True, True): 0.2,
                                             (False, True, False): 0.0,
                                             (False, False, True): 0.1,
                                             (False, False, False): 0.0})

    jailed = CPT('J', 'G', {(True,): 0.9, (False,): 0.0})

    # store the topologically ordered list of variables/objects
    # traverse this list in this order to build a sample

    orderedVars = [brokeElectionLaw, politcallyMotivatedProsecutor, indicted, foundGuilty, jailed]

    # example of specifying an event to compute for direct sampling
    eventToCalc = {'B': True, 'M': False, 'I': True, 'G': True, 'J': True}

    runningEstimate_B_notM_I_G_J = directSampling(orderedVars, eventToCalc, 1000)
    print('final estimate of this event is:{:.4f}'.format(runningEstimate_B_notM_I_G_J[-1]))

    plotRunningEstimate(runningEstimate_B_notM_I_G_J, 'runningEstimate_B_notM_I_G_J.pdf', eventToCalc)

    orderedVars = [brokeElectionLaw]
    eventToCalc = {'B': True}

    runningEstimate_B = directSampling(orderedVars, eventToCalc, 1000)
    print('final estimate of this event is:{:.4f}'.format(runningEstimate_B[-1]))

    plotRunningEstimate(runningEstimate_B, 'runningEstimate_B.pdf', eventToCalc)

    # example of using rejection sampling
    orderedVars = [brokeElectionLaw, politcallyMotivatedProsecutor, indicted, foundGuilty, jailed]

    # This call computes P(b | i)
    queryVar = {'B': True}
    evidence = {'I': True}
    nbrSamplesRequested = 100000

    rs_b_given_i = rejectSampling(orderedVars, queryVar, evidence, nbrSamplesRequested, 2)
    print('reject sampling rejected:', rs_b_given_i[0] * 100, '% of the samples for not matching the evidence')
    print('final estimate of P(b | i) is:', rs_b_given_i[1][-1])

    plotRunningEstimate(rs_b_given_i[1], 'rs_b_given_i.pdf', queryVar, evidence)

    queryVar = {'M': True}
    evidence = {'J': True, 'B': False}
    nbrSamplesRequested = 100000

    rs_m_given_jb = rejectSampling(orderedVars, queryVar, evidence, nbrSamplesRequested, 584.795322)
    print('reject sampling rejected:', rs_m_given_jb[0] * 100, '% of the samples for not matching the evidence')
    print('final estimate of P(b | i) is:', rs_m_given_jb[1][-1])

    plotRunningEstimate(rs_m_given_jb[1], 'rs_m_given_jb.pdf', queryVar, evidence)


if __name__ == "__main__":
    main()
