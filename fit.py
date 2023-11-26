from ddHCRP_LM import *

# UCRP_LM and SCRP_LM are candidate models requested by the Reviewers; not used here
# from UCRP_LM import *
# from SCRP_LM import *

import _pickle as cPickle
import os, sys, inspect
import psutil
from scipy.stats import truncnorm

def fit(subject, iteration):

    np.random.seed(iteration)

    s_i = "{subject}_{iteration}".format(subject = int(subject), iteration = int(iteration))

    try:
        with open(r"{s_i}_log.pickle".format(s_i=s_i), "rb") as input_file:
            logfile = cPickle.load(input_file)
            starting_offline_dist = max(logfile.keys())
            print("Subject {subject}, iteration {iteration} was started before. Logfile loaded.".format(subject = subject, iteration = iteration), flush=True)

            try:
                results = pd.read_csv('{subject}_{iteration}.csv'.format(subject=subject, iteration = iteration))
                print("Subject {subject}, iteration {iteration} posteriors from previous sessions loaded.".format(subject = subject, iteration = iteration), flush=True)

            except FileNotFoundError:
                print("Subject {subject}, iteration {iteration} posteriors from previous sessions not found. Only current posteriors will be in the output.".format(subject = subject, iteration = iteration), flush=True)
                results = pd.DataFrame()

    except IOError:

        logfile = {}
        starting_offline_dist = 0
        results = pd.DataFrame()

        print("Subject {subject}, iteration {iteration} is started for the first time.".format(subject = subject, iteration = iteration), flush=True)

    subject_df = df[(df['Subject'] == int(subject))].reset_index(drop=True)

    #for offline_dist in range(starting_offline_dist, 100, 33):
    for offline_dist in [0]:

        if offline_dist in logfile.keys():
            starting_session = logfile[offline_dist] + 1  # starting from after last session that was fitted for this subject, iteration and offline distance value
            if starting_session > (len(subject_df.Session.unique())+1):
                print("All sessions of subject {subject}, iteration {iteration},\
                        offline distance {offline_dist} were fitted already.\
                        Jumping to next setting.".format(subject       = subject,
                                                            iteration   = iteration,
                                                            offline_dist= offline_dist),
                        flush=True
                        )
                continue  # all sessions were fitted already, proceed to next offline distance value
        else:
            starting_session = 1

        for session in range(starting_session, len(subject_df.Session.unique())+1):

            print("Starting subject {subject}, iteration {iteration}, offline dist {offline_dist}, session {session}".format(subject    = subject,
                                                                                                                            iteration   = iteration,
                                                                                                                            offline_dist= offline_dist,
                                                                                                                            session     = session),
                                                                                                                            flush=True)
            min_NLL = 10**10
            subdf = subject_df[(subject_df['Session'] == session)]
            if len(subdf) == 0:
                continue  # in case session is missing from the subject

            outlier_mask = np.where(np.abs(scipy.stats.zscore(np.log(subdf['measured_RT']))) > 3, False, True)
            practice_mask = np.where(subdf['TrialType']=='Prac', False, True)
            training_mask = epoch_training_mask if len(subdf)==425 else session_training_mask
            mask = np.logical_and.reduce([training_mask, outlier_mask, practice_mask])

            t_start = subdf.index[0]

            if session <= 5:

                for param_search_iter in range(n_param_search_iter):

                    m = UCRP_LM(
                                # strength       = list(np.random.uniform(lower_strength, upper_strength, 4)),
                                #decay_constant = list(np.random.uniform(start_lower_decay, start_upper_decay, 4))
                                strength       = [np.random.uniform(lower_strength, upper_strength)]*4,
                                decay_constant = [np.random.uniform(start_lower_decay, start_upper_decay)]*4  # uniform forgetting
                                )

                    response_model, predicted_RT, NLL = compute_NLL_of_HCRP(m                    = m,
                                                                            t_start              = t_start,
                                                                            subdf                = subdf,
                                                                            lowlevel_predictors  = lowlevel_predictors,
                                                                            mask                 = mask,
                                                                            offline_dist         = offline_dist,
                                                                            resp_noise           = resp_noise,
                                                                            frozen               = frozen)

                    if NLL < min_NLL:

                        min_NLL                 = NLL
                        best_model              = copy.deepcopy(m)
                        best_response_params    = [response_model.intercept_] + list(response_model.coef_)

            else:

                for param_search_iter in range(n_param_search_iter):

                    with open(r"model_{s_i}.pickle".format(s_i = s_i), "rb") as input_file:
                         m = cPickle.load(input_file)

                    # m = copy.deepcopy(current_fitted_model)  # used when not checkpointing but saving the model as variable

                    strength_previous, decay_previous =  np.array(m.strength), np.array(m.decay_constant)

                    # m.strength       =  truncnorm((lower_strength - strength_previous) / sigma_strength, (upper_strength - strength_previous) / sigma_strength, loc=strength_previous, scale=sigma_strength).rvs().tolist()
                    # m.decay_constant =  truncnorm((lower_decay - decay_previous) / sigma_decay, (upper_decay - decay_previous) / sigma_decay, loc=decay_previous, scale=sigma_decay).rvs().tolist()


                    strength_previous = strength_previous[0]
                    m.strength       =  [truncnorm((lower_strength - strength_previous) / sigma_strength, (upper_strength - strength_previous) / sigma_strength, loc=strength_previous, scale=sigma_strength).rvs()] * 4


                    decay_previous = decay_previous[0]
                    m.decay_constant =  [truncnorm((lower_decay - decay_previous) / sigma_decay, (upper_decay - decay_previous) / sigma_decay, loc=decay_previous, scale=sigma_decay).rvs()] * 4

                    response_model, predicted_RT, NLL = compute_NLL_of_HCRP(m                    = m,
                                                                            t_start              = t_start,
                                                                            subdf                = subdf,
                                                                            lowlevel_predictors  = lowlevel_predictors,
                                                                            mask                 = mask,
                                                                            offline_dist         = offline_dist,
                                                                            resp_noise           = resp_noise,
                                                                            frozen               = frozen)

                    if NLL < min_NLL:

                        min_NLL                 = NLL
                        best_model              = copy.deepcopy(m)
                        best_response_params    = [response_model.intercept_] + list(response_model.coef_)


            logfile[offline_dist] = session
            with open(r"{s_i}_log.pickle".format(s_i=s_i), "wb") as output_file:
                 cPickle.dump(logfile, output_file)

            # current_fitted_model = copy.deepcopy(best_model)  # used when not checkpointing but saving the model as variable

            with open(r"model_{s_i}.pickle".format(s_i=s_i), "wb") as output_file:
                 cPickle.dump(best_model, output_file)

            results = results.append(pd.Series([subject, iteration, session, offline_dist]\
                                                + copy.deepcopy(best_model.strength)\
                                                + copy.deepcopy(best_model.decay_constant) \
                                                + best_response_params\
                                                + [min_NLL]),
                                                ignore_index = True)
            results.to_csv('{subject}_{iteration}.csv'.format(subject=subject, iteration = iteration), index = False)

    return results

################################################################################

df = pd.read_csv('data.csv', dtype={'choice': str})

### SETTINGS FOR MAIN RESULTS
# start_lower_decay, start_upper_decay        = 1, 80
# lower_strength, upper_strength, sigma_strength  = 0.00001, 50, 50
# lower_decay, upper_decay, sigma_decay           = 1, 2000, 1000

### SETTINGS FOR ERROR RESULTS (MORE FORGETFUL REGIME)
start_lower_decay, start_upper_decay        = 1, 80
lower_strength, upper_strength, sigma_strength  = 0.00001, 50, 50
lower_decay, upper_decay, sigma_decay           = 1, 80, 80

session_training_mask = [True] * 11 * 85 + [False] * 3 * 85 + [True] * 11 * 85
epoch_training_mask = [True] * 2 * 85 + [False] * 1 * 85 + [True] * 2 * 85
lowlevel_predictors = ['spatial_distance', 'repetition', 'error', 'posterror']
frozen              = False
resp_noise          = 0.1  ## 0.2
n_param_search_iter = 1000
iterations          = range(2)
subjects            = list(df.Subject.unique())
print(subjects)
subjects_iterations = [(s, i) for s in subjects for i in iterations]
print(subjects_iterations)

Session = []
for i,r in df.iterrows():
    if i%10000==0: print(i)
    if r['Session'] == 1:
        Session.append(r['epoch'])
    elif r['Session'] > 1 and r['Session'] < 9:
        Session.append(r['Session']+4)
    elif r['Session'] == 9:
        Session.append(12 + r['epoch'])
    else:
        Session.append(17 + r['epoch'])
df['Session'] = Session
df = df[df['Session']<=12]

import multiprocessing as mp
if __name__ == '__main__':
    #n_physical_cores = psutil.cpu_count(logical = False)
    n_physical_cores = 64  # hardcoded for the MPI HP cluster
    print('n_physical_cores :' + str(n_physical_cores))
    print('len(subjects_iterations):' + str(len(subjects_iterations)))
    pool = mp.Pool( min( len(subjects_iterations), n_physical_cores ) )
    print('Pool created.')
    results = pool.starmap(fit, subjects_iterations)  # starmap is used because we have two arguments. It accepts a sequence of argument tuples, then automatically unpacks the arguments from each tuple and passes them to the given function.
    pool.close()
    results = pd.concat(results)
    results.columns = ['subject',\
                        'iteration',\
                        'session',\
                        'offline_dist'] \
                        + ['best_strength_'+ str(x) for x in range(1,5)]  \
                        + ['best_decayconstant_'+ str(x) for x in range(1,5)] \
                        + ['intercept', 'spatial_distance_coef', 'repetition_coef', 'error_coef', 'posterror_coef', 'prob_coef'] \
                        + ['NLL']

    results.to_csv('posterior_values_UCRP.csv')
