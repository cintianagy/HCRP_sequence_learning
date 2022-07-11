import os, sys
import copy
import numpy as np
from random import random as randomvalue_0_1
e=np.e
import math
import scipy
from scipy.stats import pearsonr
import time
import pandas as pd
pd.options.mode.chained_assignment = None  # silence SettingWithCopyWarning; default='warn'
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#sns.set(style="white",context='paper',rc={"lines.linewidth": 1.5})
#np.set_printoptions(suppress=True)
sns.set(style="white",context='paper',font_scale=2)

from utils import *

class HCRP_LM:

    def __init__(self, strength, decay_constant=False, n_samples=5):
        self.description = "Hierarchical chinese restaurant process language model based on Yee Whye Teh (2006)"
        self.strength = strength
        self.n = len(strength) # order+1; 0th-order Markov model corresponds to 1-gram

        self.rest_labels=[]

        self.n_samples = n_samples
        self.samples = dict()
        for sample in range(n_samples):
            self.samples[sample] = dict()

        # if decay_constant == None:
        #     self.decay_constant = [np.inf]*len(self.strength)
        # else:
        #     self.decay_constant = decay_constant
        self.decay_constant = decay_constant


    def __repr__(self):
        return ('HCRP(n={self.n}, strength={self.strength}, decay_constant={self.decay_constant})').format(self=self)


    def word_probability(self, t, u, w, sample, n=None, seat_odds=False):
        """
        Returns the probability that the next word after context u will be w.
        We need the auxiliary n variable to track the n-gram levels (context length
        doesn't serve the same purpose, because we want to continue the recursion even at
        context length 0 and finish after that).
        """

        if w not in self.dishes:
            self.dishes.append(w)
            self.number_of_dishes += 1

        w_i = self.dishes.index(w)

        if n is None:
            n = len(u)+1

            if seat_odds:
                seat_odds = np.zeros(n)

        if n == 0:
            return (1 / self.number_of_dishes, seat_odds)  # G_0 prior: global mean vector with a uniform value of 1/vocabulary_size

        # no restaurant yet:
        if str(u) not in self.samples[sample].keys():

            d_u, d_u_w = 0,0

            if self.decay_constant:
                self.samples[sample][str(u)] = np.full((self.number_of_dishes, 1000), np.nan)

            else:
                self.samples[sample][str(u)] = np.zeros(self.number_of_dishes)

            self.rest_labels.append(str(u))

        # no table yet
        elif self.samples[sample][str(u)].shape[0] < w_i:

            d_u_w = 0

            if self.decay_constant:

                self.samples[sample][str(u)] = np.vstack(self.samples[sample][str(u)], np.full(1000, np.nan))

                timestamps_u = self.samples[sample][str(u)][~np.isnan(self.samples[sample][str(u)])].ravel()
                distances_u = t - timestamps_u
                decay_constant = self.decay_constant[len(u)]
                d_u = np.sum(e**(-distances_u/decay_constant))

            else:

                self.samples[sample][str(u)] = np.append(self.samples[sample][str(u)], 0)

                d_u = self.samples[sample][str(u)].sum()

        else:

            if self.decay_constant:

                timestamps_u = self.samples[sample][str(u)][~np.isnan(self.samples[sample][str(u)])].ravel()
                timestamps_u_w = self.samples[sample][str(u)][w_i][~np.isnan(self.samples[sample][str(u)][w_i])]

                distances_u = t - timestamps_u
                decay_constant = self.decay_constant[len(u)]
                d_u = np.sum(e**(-distances_u/decay_constant))

                distances_u_w = t - timestamps_u_w
                d_u_w = np.sum(e**(-distances_u_w/decay_constant))

            else:
                d_u     = self.samples[sample][str(u)].sum()
                d_u_w   = self.samples[sample][str(u)][w_i]

        strength_u = self.strength[len(u)]

        prob_seat       = (d_u_w / (d_u+strength_u))
        prob_backoff    = (strength_u / (d_u+strength_u)) * self.word_probability(t, u[1:], w, sample, n-1, seat_odds)[0]
        prob = prob_seat + prob_backoff

        if type(seat_odds) is np.ndarray:
            seat_odds[n-1] = prob_seat/prob_backoff

        return (prob, seat_odds)

    def word_probability_all_samples(self, t, u, w):
        sum_word_probabilities = 0
        for sample in self.samples.keys():
            sum_word_probabilities += self.word_probability(t, u, w, sample)[0]
        return sum_word_probabilities/len(self.samples.keys())

    #@profile
    def add_customer(self, t, u, w, sample, n=None):

        if w not in self.dishes:
            self.dishes.append(w)
            self.number_of_dishes += 1

        w_i = self.dishes.index(w)

        if n is None:
            n = len(u)+1

        if n==0:
            return

        else:

            # no restaurant yet:
            if str(u) not in self.samples[sample].keys():
                d_u_w = 0
                if self.decay_constant:
                    self.samples[sample][str(u)] = np.full((self.number_of_dishes, 1000), np.nan)
                    self.samples[sample][str(u)][w_i][0] = t
                else:
                    self.samples[sample][str(u)] = np.zeros(self.number_of_dishes)
                    self.samples[sample][str(u)][w_i] = 1
                self.rest_labels.append(u)

            # no table yet
            elif self.samples[sample][str(u)].shape[0] < w_i:

                d_u_w = 0

                if self.decay_constant:

                    self.samples[sample][str(u)] = np.vstack(self.samples[sample][str(u)], np.full(1000, np.nan))
                    self.samples[sample][str(u)][w_i][0] = t

                    timestamps_u = self.samples[sample][str(u)][~np.isnan(self.samples[sample][str(u)])].ravel()
                    distances_u = t - timestamps_u
                    decay_constant = self.decay_constant[len(u)]
                    d_u = np.sum(e**(-distances_u/decay_constant))

                else:

                    self.samples[sample][str(u)] = np.append(self.samples[sample][str(u)], 0)
                    self.samples[sample][str(u)][w_i] = 1

                    d_u = self.samples[sample][str(u)].sum()

            else:

                if self.decay_constant:

                    timestamps_u_w = self.samples[sample][str(u)][w_i][~np.isnan(self.samples[sample][str(u)][w_i])]
                    decay_constant = self.decay_constant[len(u)]

                    # no table with this dish yet:
                    if not len(timestamps_u_w):
                        d_u_w = 0

                    else:
                        distances_u_w = t - timestamps_u_w
                        d_u_w = np.sum(e**(-distances_u_w/decay_constant))

                    if len(timestamps_u_w)<1000:
                        self.samples[sample][str(u)][w_i][len(timestamps_u_w)] = t
                    else:
                        self.samples[sample][str(u)][w_i] = np.append(self.samples[sample][str(u)][w_i][1:], t) # if 1000 values stored, drop one oldest

                else:
                    d_u_w = self.samples[sample][str(u)][w_i]
                    self.samples[sample][str(u)][w_i] += 1


            # choose to backoff
            unnormalized_probs = [d_u_w] + [self.strength[len(u)] * self.word_probability(t, u[1:], w, sample, n-1)[0]]
            normalized_prob_of_seating_at_old = d_u_w/sum(unnormalized_probs)

            # seated at existing table -> return
            if normalized_prob_of_seating_at_old > np.random.rand(): return

            # opened new table -> backoff
            else:
                self.add_customer(t, u[1:], w, sample, n-1)  # backoff
                return


    def fit_onesample(self, sample, corpus_segments, choices_segments=False, t_start=0, online_predict=False, compute_seat_odds=False, compute_context_importance=False, frozen=True):

        if frozen:

            t_end = 0
            for i_segment in range(len(corpus_segments)):
                print(i_segment)
                corpus, choices = corpus_segments[i_segment], choices_segments[i_segment]

                for t in range(len(corpus)):

                    t_g = t+t_end  # t_global

                    u, w, choice = corpus[max(0, t - self.n + 1):t], corpus[t], choices[t]
                    self.add_customer(t_g+t_start, u, w, sample)

                    # print(self.samples)
                    # print('---------------------------------------------------')

                t_end = t_g+1

            t_end = 0
            for i_segment in range(len(corpus_segments)):
                corpus, choices = corpus_segments[i_segment], choices_segments[i_segment]

                for t in range(len(corpus)):

                    t_g = t+t_end  # t_global

                    u, w, choice = corpus[max(0, t - self.n + 1):t], corpus[t], choices[t]
                    choice_prob, seat_odds = self.word_probability(t=t_g+t_start, u=u, w=choice, sample=sample, n=None, seat_odds=compute_seat_odds)

                    self.sample_choice_probs[sample][t_g] = choice_prob

                    if compute_seat_odds:
                        self.sample_seat_odds[sample][t_g][:len(u)+1] = seat_odds

                    if online_predict:
                        for w in self.dishes:
                            self.sample_predictive_distr[sample][t_g][self.dishes.index(w)], seat_odds = self.word_probability(t_g+t_start, u, w, sample)

                        if compute_context_importance:
                            for context_len in range(self.n):
                                word_probs_given_context_len = np.zeros(self.number_of_dishes)

                                for w in self.dishes:
                                    word_probs_given_context_len[self.dishes.index(w)], seat_odds = self.word_probability(t_g+t_start, u[-context_len:], w, sample)

                                KL_div = scipy.stats.entropy(word_probs_given_context_len, self.sample_predictive_distr[sample][t_g])
                                self.sample_context_importance[sample][t_g][context_len] = KL_div

                t_end = t_g+1

        else:

            t_end = 0
            for i_segment in range(len(corpus_segments)):
                corpus, choices = corpus_segments[i_segment], choices_segments[i_segment]

                for t in range(len(corpus)):

                    t_g = t+t_end  # t_global

                    u, w, choice = corpus[max(0, t - self.n + 1):t], corpus[t], choices[t]
                    choice_prob, seat_odds = self.word_probability(t=t_g+t_start, u=u, w=choice, sample=sample, n=None, seat_odds=compute_seat_odds)

                    self.sample_choice_probs[sample][t_g] = choice_prob

                    if compute_seat_odds:
                        self.sample_seat_odds[sample][t_g] = seat_odds

                    if online_predict:
                        for w in self.dishes:
                            self.sample_predictive_distr[sample][t_g][self.dishes.index(w)], seat_odds = self.word_probability(t_g+t_start, u, w, sample)

                        if compute_context_importance:
                            for context_len in range(self.n):
                                word_probs_given_context_len = np.zeros(self.number_of_dishes)

                                for w in self.dishes:
                                    context = u[-context_len:] if context_len>0 else ''
                                    word_probs_given_context_len[self.dishes.index(w)], seat_odds = self.word_probability(t_g+t_start, context, w, sample)

                                KL_div = scipy.stats.entropy(word_probs_given_context_len, self.sample_predictive_distr[sample][t_g])
                                self.sample_context_importance[sample][t_g][context_len] = KL_div

                    self.add_customer(t_g+t_start, u, w, sample)

                t_end = t_g+1

    def fit(self, corpus_segments, choices_segments=False, t_start=0, online_predict=False, compute_seat_odds=False, compute_context_importance=False, frozen=True):

        if not any(isinstance(i, list) for i in corpus_segments):
            corpus_segments = [corpus_segments]

        if choices_segments==False:
            choices_segments = corpus_segments
        else:
            if not any(isinstance(i, list) for i in choices_segments):
                choices_segments = [choices_segments]


        self.dishes = sorted(list(set([element for segment in corpus_segments for element in segment])))
        self.number_of_dishes = len(self.dishes)

        self.sample_choice_probs = np.zeros((self.n_samples, len(flatten(corpus_segments))))
        if online_predict:
            self.sample_predictive_distr = np.zeros((self.n_samples, len(flatten(corpus_segments)), self.number_of_dishes))
            if compute_context_importance:
                self.sample_context_importance = np.zeros((self.n_samples, len(flatten(corpus_segments)), self.n))
        if compute_seat_odds:
            self.sample_seat_odds = np.zeros((self.n_samples, len(flatten(corpus_segments)), self.n))

        for sample in self.samples.keys():
            self.fit_onesample( sample                      = sample,
                                corpus_segments             = corpus_segments,
                                choices_segments            = choices_segments,
                                t_start                     = t_start,
                                online_predict              = online_predict,
                                compute_seat_odds           = compute_seat_odds,
                                compute_context_importance  = compute_context_importance,
                                frozen                      = frozen)

        self.choice_probs = np.mean(self.sample_choice_probs, axis=0)

        if online_predict:

            self.predictive_distr = np.mean(self.sample_predictive_distr, axis=0)
            index_of_most_likely_events = np.argmax(self.predictive_distr, axis=1)
            self.event_predictions = np.zeros(len(flatten(corpus_segments)), dtype='str')
            for i, dish in enumerate(self.dishes): self.event_predictions[index_of_most_likely_events==i] = dish

            if compute_context_importance:
                self.context_importance = np.mean(self.sample_context_importance, axis=0)

        if compute_seat_odds:
            self.seat_odds = np.mean(self.sample_seat_odds, axis=0)

    def negLL(self):
        return -np.sum(np.log(np.array(self.choice_probs)))


################################################################################

def fit_response_model(model_predictive_probabilities, lowlevel_predictors, subdf, mask=None, return_predicted_RT_components=False):

    if mask is None: mask = np.array([True]*len(subdf))

    X            = np.hstack((subdf[lowlevel_predictors].values,
                                -np.log(model_predictive_probabilities).reshape(-1,1)))
    y            = subdf.measured_RT.values

    response_model = LinearRegression(fit_intercept=True).fit(X[mask], y[mask])
    predicted_RTs = response_model.predict(X).ravel()

    if return_predicted_RT_components:

        predicted_RT_components = response_model.coef_ * X
        intercept = np.full(X.shape[0], response_model.intercept_).reshape(-1, 1)
        predicted_RT_components = np.append(intercept, predicted_RT_components, axis=1)

        return response_model, predicted_RTs, predicted_RT_components

    else:
        return response_model, predicted_RTs

def compute_NLL_of_HCRP(m, t_start, subdf, lowlevel_predictors, mask, offline_dist, resp_noise, frozen=True):

    corpus_segments  = list(subdf.event.astype('str'))
    choices_segments = list(subdf.choice.astype('str'))
    session          = subdf.Session.iloc[0]

    m.fit(t_start            = t_start,
            corpus_segments  = corpus_segments,
            choices_segments = choices_segments,
            frozen           = frozen)

    HCRP_response_model, HCRP_predicted_RT    = fit_response_model(model_predictive_probabilities = m.choice_probs,
                                                                    lowlevel_predictors           = lowlevel_predictors,
                                                                    subdf                         = subdf,
                                                                    mask                          = mask)

    # We evaluate the log likelihoods of the log RTs because we want less of a heavy tail, assuming Gaussian noise in our likelihood function
    NLL = compute_NLL(np.log(subdf.measured_RT.values), np.log(HCRP_predicted_RT), resp_noise, mask)

    return HCRP_response_model, HCRP_predicted_RT, NLL
