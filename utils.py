import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def flatten(t):
    return [item for sublist in t for item in sublist]

def corrfunc(x,y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot - like seaborn pairplot."""
    """You will need to call g.map_lower(corrfunc) """
    r, p = pearsonr(x, y)
    ax = ax or plt.gca()
    if p<.001:
        ax.annotate('r=' + str(np.round(r,2)) + '; p<.001', xy=(.1, .9), weight='bold', bbox=dict(facecolor='red', alpha=0.5), xycoords=ax.transAxes)
    elif p<.05:
        ax.annotate('r=' + str(np.round(r,2)) + '; p=' + str(np.round(p,3)), xy=(.1, .9), weight='bold', bbox=dict(facecolor='red', alpha=0.5), xycoords=ax.transAxes)
    else:
        ax.annotate('r=' + str(np.round(r,2)) + '; p=' + str(np.round(p,3)), xy=(.1, .9), xycoords=ax.transAxes)

def min_max_scale(a, lim_left=0, lim_right=1):
    scaled = lim_left + ((lim_right - lim_left) / (a.max() - a.min())) * (a-a.min())
    if a.min() < 0:
        return scaled - scaled.min() # special case if distribution has negative values
    else:
        return scaled

def shannon_entropy(events):
    return scipy.stats.entropy(pd.Series(events).value_counts())

def softmax(a):
    return np.exp(a)/sum(np.exp(a))

# def my_normpdf(x, mean, sd):
#     var = float(sd)**2
#     denom = (2*np.pi*var)**.5
#     num = np.exp(-(float(x)-float(mean))**2/(2*var))
#     return num/denom

def my_normpdf(xs, means, sd):
    #vector function
    var = sd**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(np.array(xs)-np.array(means))**2/(2*var))
    return num/denom

def compute_NLL(xs, means, sd, mask):

    LLs = my_normpdf(np.array(xs)[mask], np.array(means)[mask], sd)

    return -np.sum(LLs)

def unigram_stim(T):
    # generate sequence with biased event frequency - so that the unigram level is learnable
    return ''.join(np.random.choice(a = ['1','2','3','4'], size = T, p = [0.1,0.1,0.1,0.7]))

def bigram_SRT_stim(T):
    # one frequent ngram interleaved by random blocks of size n
    pattern = ''.join(np.random.choice(a = ['1','2','3','4'], size = 10))
    s = ''
    trialtype = 'P'
    while len(s)<T:
        if trialtype == 'P':
            s += pattern
            trialtype = 'R'
        else:
            s += ''.join(np.random.choice(a = ['1','2','3','4'], size = 1))
            trialtype = 'P'
    return s

def fivegram_SRT_stim(T):
    # one frequent ngram interleaved by random blocks of size n
    pattern = ''.join(np.random.choice(a = ['1','2','3','4'], size = 5))
    s = ''
    trialtype = 'P'
    while len(s)<T:
        if trialtype == 'P':
            s += pattern
            trialtype = 'R'
        else:
            s += ''.join(np.random.choice(a = ['1','2','3','4'], size = 5))
            trialtype = 'P'
    return s

def robertson_2007_SRT_stim(T):
    pattern = ['2','3','1','4','3','2','4','1','3','4','2','1']
    return ''.join(pattern*int(np.ceil((T/len(pattern)))))[:T]

def ASRT_stim(T,E=['1','2','3','4'], return_trialtypes=False):

    second_order_pairs = [('0', '1'), ('1', '2'), ('2', '3'), ('3', '0')]

    i=0
    trialtype = 'P'
    pattern_i = 0
    S = []
    trialtypes      = []
    triplettypes    = []
    while i < T:

        if trialtype == 'P':
            S.append(E[pattern_i])
            trialtypes.append(trialtype)
            triplettypes.append('H')

            if pattern_i<len(E)-1:
                pattern_i += 1
            else:
                pattern_i = 0
            trialtype = 'R'

        else:
            S.append(np.random.choice(E))
            trialtypes.append(trialtype)
            if len(S)>2 and (S[-3], S[-1]) in second_order_pairs:
                triplettypes.append('RH')
            else:
                triplettypes.append('RL')

            trialtype = 'P'
        i+=1

    if return_trialtypes:
        return ''.join(S), trialtypes, triplettypes

    else:
        return ''.join(S)
