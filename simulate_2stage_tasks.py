# Copyright 2017 Carolina Feher da Silva <carolfsu@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Runs the simulations of two-stage tasks and analyze them."""

import pickle
import random
import os
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression

# Constants
# Number of trials
NUM_TRIALS = 250
# Common probability (the rare probability is 1 - COMMON_PROB)
COMMON_PROB = 0.7
# Common transitions
COMMON_TRANS = {
    'l': 'p', # Left to pink
    'r': 'b', # Right to blue
}
# Number of second-stage actions
NUM_FSA = 2
# File to save the simulation results (dataframes of results)
RESULTS_FN = 'sim_results.dat'

# Simulation parameters
ALPHA, LMBD, BETA, P_PARAM = 0.5, 0.6, 5, 0
# List of simulation parameters
PARAMS_LIST = (
    # Parameters:
    # - simulation number (int)
    # - alpha
    # - lambda
    # - beta
    # - p (perseveration parameter)
    # - final-state probabilities (see Exp initialization)
    # - diffuse_probs (bool)

    # Simulate task from Daw et al. 2011
    (1, ALPHA, LMBD, BETA, P_PARAM, None, True),

    # Task where all final-state actions have 0.5 reward prob
    (2, ALPHA, LMBD, BETA, P_PARAM, {
        (s, a): 0.5 for s in ('p', 'b') for a in range(NUM_FSA)}, False),

    # Task where final-state actions have 0.8/0.2 reward prob
    (3, ALPHA, LMBD, BETA, P_PARAM, {
        (s, a): (0.8 if s == 'p' else 0.2) for s in ('p', 'b')\
            for a in range(NUM_FSA)}, False),

    # Task where final-state actions have 0.8/0.2 reward prob, lambda = 1
    (4, ALPHA, 1, BETA, P_PARAM, {
        (s, a): (0.8 if s == 'p' else 0.2) for s in ('p', 'b')\
            for a in range(NUM_FSA)}, False),

    # Task where all final-state actions have 0.8 reward prob
    (5, ALPHA, LMBD, BETA, P_PARAM, {
        (s, a): 0.8 for s in ('p', 'b') for a in range(NUM_FSA)}, False),
)
# Number of simulated agents of each type (model-free, model-based)
NUM_AGENTS = 1000

def diffuse_prob(prob):
    """Diffuses a probability between 0.25 and 0.75"""
    prob += random.gauss(0, 0.025)
    if prob < 0.25:
        prob = 0.5 - prob
    elif prob > 0.75:
        prob = 1.5 - prob
    assert prob >= 0.25 and prob <= 0.75
    return prob

class Exp:
    """Creates an experiment (sequence of trials).

    The final states are 'p' (pink) and 'b' (blue).
    The actions at the first state are 'l' (left) and 'r' (right).
    The actions at the final states are 0, 1, ..., (NUM_FSA - 1).
    """
    def __init__(self, rwrd_probs=None):
        """Experiment initialization.

        Keyword argument
        rwrd_probs: reward probabilities of final-state actions, should be a
            dict with (state, action) tuples as keys and the probabilities as
            values, or None (default, determine probabilities randomly)
        """
        if rwrd_probs is None:
            # Determine initial reward probabilities randomly
            self.rwrd_probs = {
                (s, a): random.uniform(0.25, 0.75)\
                for s in ('p', 'b') for a in range(NUM_FSA)}
        else:
            self.rwrd_probs = rwrd_probs
        self.trial = 0
        self.trial_info = []
        self.common = None
        self.finalst = None
        self.choice = None
        self.reward = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.trial >= NUM_TRIALS:
            raise StopIteration
        self.common = random.random() < COMMON_PROB # Probability of common transition
        self.trial += 1
        return self.trial - 1

    def enter_choice1(self, choice):
        """Enter the initial-state choice ('l' or 'r')."""
        self.finalst = COMMON_TRANS[choice] # Common
        self.choice = choice
        if not self.common: # Rare
            if self.finalst == 'p':
                self.finalst = 'b'
            else:
                self.finalst = 'p'

    def enter_choice2(self, choice, diffuse_probs=True):
        """Enter the final-state choice.
        
        Keyword arguments
        choice: the final-state choice
        diffuse_probs: determines if probabilities should be diffused
            (default: True)
        """
        self.reward = random.random() < self.rwrd_probs[(self.finalst, choice)]
        self.trial_info.append({
            'trial': self.trial + 1,
            'common': int(self.common),
            'choice': self.choice,
            'finalst': self.finalst,
            'reward': int(self.reward),
        })
        if diffuse_probs:
            # Diffuse probability
            for k, v in self.rwrd_probs.items():
                self.rwrd_probs[k] = diffuse_prob(v)

    def get_results(self):
        """Get dataframe with the results of the experiment."""
        cols = self.trial_info[0].keys()
        results = pd.DataFrame(columns=cols)
        results.trial = results.trial.astype('int')
        results.common = results.common.astype('int')
        results.reward = results.reward.astype('int')

        for trial_num, info in enumerate(self.trial_info):
            results.loc[trial_num] = pd.Series(info)
        return results

def get_sschoice(q2, beta, finst):
    """Get simulated choice.

    Keyword arguments:
    q2: dict of final-state action values
    beta: exploration parameter
    finst: final state
    """
    probs = np.array(
        [np.exp(beta*v) for (s, a), v in q2.items() if s == finst])
    probs /= np.sum(probs)
    r = random.random()
    s = 0
    for action, x in enumerate(probs):
        s += x
        if s >= r:
            return action
    return action

def model_free_sim(alpha, lmbd, beta, p, rwrd_probs, diffuse_probs):
    """Simulates a model-free agent."""
    q1 = {}
    q1['l'] = 0
    q1['r'] = 0
    q2 = {(s, a): 0 for s in ('p', 'b') for a in range(NUM_FSA)}

    exp = Exp(rwrd_probs)
    prev_choice = random.choice(('l', 'r'))
    for trial in exp:
        rep = 1 if prev_choice == 'l' else -1
        p_left = expit(beta*(q1['l'] - q1['r'] + rep*p))
        if random.random() < p_left:
            a1 = 'l'
        else:
            a1 = 'r'
        exp.enter_choice1(a1)
        prev_choice = a1
        finst = exp.finalst
        a2 = get_sschoice(q2, beta, finst)
        exp.enter_choice2(a2, diffuse_probs=diffuse_probs)
        r = int(exp.reward)

        q1[a1] = (1 - alpha)*q1[a1] + alpha*q2[(finst, a2)] +\
            alpha*lmbd*(r - q2[(finst, a2)])
        q2[(finst, a2)] = (1 - alpha)*q2[(finst, a2)] + alpha*r
    return exp

def model_based_sim(alpha, beta, p, rwrd_probs, diffuse_probs):
    """Simulates a model-based agent."""
    exp = Exp(rwrd_probs)
    q2 = {(s, a): 0 for s in ('p', 'b') for a in range(NUM_FSA)}
    prev_choice = random.choice(('l', 'r'))
    for trial in exp:
        vpink = max([q2[('p', a)] for a in range(NUM_FSA)])
        vblue = max([q2[('b', a)] for a in range(NUM_FSA)])
        # Determine the choice
        if COMMON_TRANS['r'] == 'p':
            cv = (2*COMMON_PROB - 1)*(vpink - vblue)
        else:
            cv = (2*COMMON_PROB - 1)*(vblue - vpink)
        rep = 1 if prev_choice == 'r' else -1
        p_right = expit(beta*(cv + rep*p))
        if random.random() < p_right:
            a = 'r'
        else:
            a = 'l'
        exp.enter_choice1(a)
        prev_choice = a
        finst = exp.finalst
        a2 = get_sschoice(q2, beta, finst)
        exp.enter_choice2(a2, diffuse_probs=diffuse_probs)
        r = int(exp.reward)
        q2[(finst, a2)] = (1 - alpha)*q2[(finst, a2)] + alpha*r
    return exp

def plot_probs(probs, legend=True):
    """Plot probabilities."""
    x = (0, 2)
    plt.bar(left=x, height=[probs[i] for i in x], align='center',
            color='tab:orange', label='common')
    x = (1, 3)
    plt.bar(left=x, height=[probs[i] for i in x], align='center',
            color='tab:green', label='rare')
    plt.xticks((0.5, 2.5), ('rewarded', 'unrewarded'))
    plt.ylabel('stay probability')
    if legend:
        plt.legend(loc='upper right', fontsize='medium')
    plt.ylim(0, 1)
    plt.xlim(-0.5, 3.5)

def get_predictors(results):
    """Get predictors for logistic regression."""
    assert len(results) == 250
    y = []
    x = []
    for (_, trial1), (_, trial2) in zip(
            results.iloc[:-1].iterrows(), results.iloc[1:].iterrows()):
        transition = 2*int(trial1.common) - 1
        reward = 2*trial1.reward - 1
        x.append([1, reward, transition, reward*transition])
        y.append(int(trial1.choice == trial2.choice))
    return x, y

def get_predictors_alt(results):
    """Get predictors for logistic regression (alternative model)."""
    assert len(results) == 250
    y = []
    x = []
    for (_, trial1), (_, trial2) in zip(
            results.iloc[:-1].iterrows(), results.iloc[1:].iterrows()):
        transition = 2*int(trial1.common) - 1
        reward = 2*trial1.reward - 1
        option = 2*int(COMMON_TRANS[trial1.choice] == 'p') - 1
        final_state = 2*int(trial1.finalst == 'p') - 1
        x.append([1, reward, transition, reward*transition, option, final_state])
        y.append(int(trial1.choice == trial2.choice))
    return x, y

def get_sim_results():
    """Run simulations and get results, or load results from file."""
    if not os.path.exists(RESULTS_FN):
        with open(RESULTS_FN, 'wb') as outf:
            for params in PARAMS_LIST:
                sim_results = []
                sim_num, alpha, lmbd, beta, p_param, rwrd_probs, diffuse_probs = params
                print('Running simulation {}...'.format(sim_num))
                for rep in range(NUM_AGENTS):
                    if rep % 100 == 0:
                        print('Creating model-free agents {}-{} of {}...'.format(
                            rep + 1, rep + 100, NUM_AGENTS))
                    exp = model_free_sim(
                        alpha, lmbd, beta, p_param, rwrd_probs, diffuse_probs)
                    results = exp.get_results()
                    sim_results.append(('Model-free', results))
                for rep in range(NUM_AGENTS):
                    if rep % 100 == 0:
                        print('Creating model-based agents {}-{} of {}...'.format(
                            rep + 1, rep + 100, NUM_AGENTS))
                    exp = model_based_sim(
                        alpha, beta, p_param, rwrd_probs, diffuse_probs)
                    results = exp.get_results()
                    sim_results.append(('Model-based', results))

                pickle.dump((sim_num, sim_results), outf)
                yield sim_num, sim_results
    else:
        with open(RESULTS_FN, 'rb') as outf:
            for params in PARAMS_LIST:
                sim_num, sim_results = pickle.load(outf)
                yield sim_num, sim_results

def run_simulations():
    """Run simulations, analyze the results, and create figures."""
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    analyses = (
        ('standard', get_predictors),
        ('alternative', get_predictors_alt),
    )
    for sim_num, sim_results in get_sim_results():
        for ana, get_predictors_f in analyses:
            figure = plt.figure()
            figure.set_size_inches(2*4, 4)
            for group_num, group in enumerate(('Model-free', 'Model-based')):
                group_results = [
                    results for g, results in sim_results if g == group]
                # Perform logistic regression with little regularization
                # although regulatization doesn't make much difference
                logreg = LogisticRegression(fit_intercept=False, C=1e6)
                x, y = [], []
                for i, results in enumerate(group_results):
                    xx, yy = get_predictors_f(results)
                    M = [0]*len(group_results)
                    M[i] = 1
                    for l in xx:
                        x.append(M + l)
                    y += yy
                logreg.fit(x, y)
                del x
                del y

                axes = plt.subplot(1, 2, group_num + 1)
                axes.spines['right'].set_color('none')
                axes.spines['top'].set_color('none')
                axes.xaxis.set_ticks_position('bottom')
                axes.yaxis.set_ticks_position('left')
                plt.title(group)
                coefs = logreg.coef_[0][
                    len(group_results):(len(group_results) + 4)]
                probs = (
                    expit(coefs[0] + coefs[1] + coefs[2] + coefs[3]),
                    expit(coefs[0] + coefs[1] - coefs[2] - coefs[3]),
                    expit(coefs[0] - coefs[1] + coefs[2] - coefs[3]),
                    expit(coefs[0] - coefs[1] - coefs[2] + coefs[3]),
                )
                plot_probs(probs)
                del group_results
                del logreg

            plt.tight_layout()
            plt.savefig(
                'results-sim{}-{}.eps'.format(sim_num, ana), bbox_inches='tight')
            plt.close()
        del sim_results

if __name__ == '__main__':
    run_simulations()
