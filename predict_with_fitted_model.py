import os, sys, inspect
from ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2)

cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# results = pd.read_csv(cwd + 'posterior_values_main.csv')
results = pd.read_csv(cwd + '/' + 'posterior_values_ddHCRP_LM.csv')
results.columns = [x.replace('best_','') for x in results.columns]
results = results[results['NLL']<10000000]
parameters = list(results.columns[5:-1])

MAP_data=[]
for subject in results.subject.unique():
    NLL_sums = pd.pivot_table(data=results[results['subject']==subject], index='iteration', aggfunc=np.sum)['NLL']
    best_iter = np.argmin(NLL_sums)
    MAP_data.append(results[(results['subject']==subject) & (results['iteration']==best_iter)])
MAP_data = pd.concat(MAP_data).drop('iteration', axis=1)

# MAP_data=[]
# for subject in results.subject.unique():
#     for session in results.session.unique():
#         posterior = results[(results['subject']==subject) & (results['session']==session)]
#         MAP_data.append(posterior[posterior['NLL']==posterior['NLL'].min()])
# MAP_data = pd.concat(MAP_data).drop('iteration', axis=1)

######################### CROSSVAL PREDICT ############################

df = pd.read_csv(cwd + '/' + 'data_101.csv', dtype={'choice':str})

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

model_predictions = {'index'              : [],
                        'HCRP_choice'     : [],
                        'HCRP_pred_prob'  : [],

                        'HCRP_pred_prob_1':   [],
                        'HCRP_pred_prob_2':   [],
                        'HCRP_pred_prob_3':   [],
                        'HCRP_pred_prob_4':   [],

                        'HCRP_context_importance_0'  : [],
                        'HCRP_context_importance_1'  : [],
                        'HCRP_context_importance_2'  : [],
                        'HCRP_context_importance_3'  : [],

                        'HCRP_seat_odds_0'  : [],
                        'HCRP_seat_odds_1'  : [],
                        'HCRP_seat_odds_2'  : [],
                        'HCRP_seat_odds_3'  : [],

                        'triplet_pred_prob' : [],
                        'new_triplet_pred_prob' : [],

                        'distance'          : [],
                        'repetition'        : [],
                        'error'             : [],
                        'HCRP'              : [],

                        'intercept_component'         : [],
                        'distance_component': [],
                        'repetition_component': [],
                        'error_component'   : [],
                        'posterror_component'   : [],
                        'HCRP_component'   : [],
                        'low-level + HCRP'  : [],

                        'triplet'           : [],
                        'low-level + triplet': [],

                        'new_triplet'           : [],
                        'low-level + new_triplet': [],

                        'train_mask'        :[],
                        'test_mask'         :[]
                        }


recency_effects = pd.DataFrame()
session_test_mask_generic       = [False] * 11 * 85 + [True] * 3 * 85 + [False] * 11 * 85
session_train_mask_generic      = [True] * 11 * 85 + [False] * 3 * 85 + [True] * 11 * 85

epoch_test_mask_generic = [False] * 2 * 85 + [True] * 1 * 85 + [False] * 2 * 85
epoch_train_mask_generic = [True] * 2 * 85 + [False] * 1 * 85 + [True] * 2 * 85

lowlevel_predictors = ['spatial_distance', 'repetition', 'error', 'posterror']

# lowlevel_coefs = pd.DataFrame()

for i, subject in enumerate(df.Subject.unique()):
    print(subject)

    subject_df = df[(df['Subject'] == int(subject))].reset_index()
    subject_MAP = MAP_data[(MAP_data['subject']==subject)]
    m = HCRP_LM(strength=[1] * 4, decay_constant=[1]*4)

    for session in subject_MAP['session'].unique():

        print(session)

        params = subject_MAP[(subject_MAP['session']==session)].squeeze()

        subdf = subject_df[(subject_df['Session'] == session)]
        outlier_mask = np.where(np.abs(scipy.stats.zscore(np.log(subdf['measured_RT']))) > 3, False, True)
        practice_mask = np.where(subdf['TrialType']=='Prac', False, True)

        corpus_segments = list(subdf.event.astype('str'))
        choices_segments = list(subdf.choice.astype('str'))

        measured_RT = subdf[['measured_RT']].values

        train_mask_generic = epoch_train_mask_generic if len(subdf)==425 else session_train_mask_generic
        test_mask_generic = epoch_test_mask_generic if len(subdf)==425 else session_test_mask_generic

        train_mask = np.logical_and.reduce([train_mask_generic, outlier_mask, practice_mask])
        test_mask = np.logical_and.reduce([test_mask_generic, outlier_mask, practice_mask])

        # PREDICT FROM HddCRP-LM

        strength = list(params[['strength_'+str(x) for x in range(1,5)]])
        decay_constant = list(params[['decayconstant_'+str(x) for x in range(1,5)]])
        offline_dist = params['offline_dist']
        m.strength          = strength
        m.decay_constant    = decay_constant

        t_start = subdf.index[0]

        m.fit(t_start                       = t_start,
                corpus_segments             = corpus_segments,
                choices_segments            = choices_segments,
                compute_context_importance  = True,
                compute_seat_odds           = True,
                online_predict              = True,
                frozen                      = False)

        model_predictions['HCRP_choice'] += list(m.event_predictions)
        model_predictions['HCRP_pred_prob'] += list(m.choice_probs)
        model_predictions['triplet_pred_prob'] += list(subdf.high_triplet_choice.replace({0:.125, 1:.625}).values)
        model_predictions['new_triplet_pred_prob'] += list(subdf.new_high_triplet_choice.replace({0:.125, 1:.625}).values)

        for level in range(m.n):
            model_predictions['HCRP_context_importance_' + str(level)] += list(m.context_importance[:,level])

        for level in range(m.n):
            model_predictions['HCRP_seat_odds_' + str(level)] += list(m.seat_odds[:,level])

        for event in m.dishes:
            model_predictions['HCRP_pred_prob_' + event] += list(m.predictive_distr[:, int(event)-1])

        HCRP_response_model, HCRP_predicted_RT, HCRP_predicted_RT_components                                      = fit_response_model(model_predictive_probabilities=m.choice_probs,
                                                                                                                                        lowlevel_predictors=[],
                                                                                                                                        subdf=subdf,
                                                                                                                                        mask=train_mask,
                                                                                                                                        return_predicted_RT_components=True)

        lowlevel_HCRP_response_model, lowlevel_HCRP_predicted_RT, lowlevel_HCRP_predicted_RT_components           = fit_response_model(model_predictive_probabilities=m.choice_probs,
                                                                                                                                        lowlevel_predictors=lowlevel_predictors,
                                                                                                                                        subdf=subdf,
                                                                                                                                        mask=train_mask,
                                                                                                                                        return_predicted_RT_components=True)

        triplet_response_model, triplet_predicted_RT, triplet_predicted_RT_components                             = fit_response_model(model_predictive_probabilities=subdf.high_triplet_choice.replace({0:.375, 1:.625}).values,
                                                                                                                                        lowlevel_predictors=[],
                                                                                                                                        subdf=subdf,
                                                                                                                                        mask=train_mask,
                                                                                                                                        return_predicted_RT_components=True)

        lowlevel_triplet_response_model, lowlevel_triplet_predicted_RT, lowlevel_triplet_predicted_RT_componenets = fit_response_model(model_predictive_probabilities=subdf.high_triplet_choice.replace({0:.375, 1:.625}).values,
                                                                                                                                        lowlevel_predictors=lowlevel_predictors,
                                                                                                                                        subdf=subdf,
                                                                                                                                        mask=train_mask,
                                                                                                                                        return_predicted_RT_components=True)

        if session>12:

            new_triplet_response_model, new_triplet_predicted_RT, new_triplet_predicted_RT_components                             = fit_response_model(model_predictive_probabilities=subdf.new_high_triplet_choice.replace({0:.375, 1:.625}).values,
                                                                                                                                            lowlevel_predictors=[],
                                                                                                                                            subdf=subdf,
                                                                                                                                            mask=train_mask,
                                                                                                                                            return_predicted_RT_components=True)

            lowlevel_new_triplet_response_model, lowlevel_new_triplet_predicted_RT, lowlevel_new_triplet_predicted_RT_componenets = fit_response_model(model_predictive_probabilities=subdf.new_high_triplet_choice.replace({0:.375, 1:.625}).values,
                                                                                                                                            lowlevel_predictors=lowlevel_predictors,
                                                                                                                                            subdf=subdf,
                                                                                                                                            mask=train_mask,
                                                                                                                                            return_predicted_RT_components=True)

        model_predictions['intercept_component']    += list(lowlevel_HCRP_predicted_RT_components[:, 0])
        model_predictions['distance_component']     += list(lowlevel_HCRP_predicted_RT_components[:, 1])
        model_predictions['repetition_component']   += list(lowlevel_HCRP_predicted_RT_components[:, 2])
        model_predictions['error_component']        += list(lowlevel_HCRP_predicted_RT_components[:, 3])
        model_predictions['posterror_component']    += list(lowlevel_HCRP_predicted_RT_components[:, 4])
        model_predictions['HCRP_component']         += list(lowlevel_HCRP_predicted_RT_components[:, 5])

        #######################################################

        # lowlevel_coefs = lowlevel_coefs.append(pd.Series(list(lowlevel_HCRP_response_model.coef_) +['HCRP']), ignore_index=True)
        # lowlevel_coefs = lowlevel_coefs.append(pd.Series(list(lowlevel_triplet_response_model.coef_) +['triplet']), ignore_index=True)

        ########################################################

        distance_model = LinearRegression(fit_intercept=True).fit(
                                            subdf[['spatial_distance']].values[train_mask & (subdf['spatial_distance']>0)],
                                            measured_RT[train_mask & (subdf['spatial_distance']>0)]
                                                                            )
        distance_predicted_RT = distance_model.predict(subdf[['spatial_distance']].values).ravel()

        #
        repetition = LinearRegression(fit_intercept=True).fit(
                                    subdf[['repetition']][train_mask].values,
                                    measured_RT[train_mask]
                                                                    )
        repetition_predicted_RT = repetition.predict(subdf[['repetition']].values).ravel()

        #
        error_model = LinearRegression(fit_intercept=True).fit(
                                            subdf[['error','posterror']].values[train_mask],
                                            measured_RT[train_mask]
                                                                            )
        error_predicted_RT = error_model.predict(subdf[['error','posterror']].values).ravel()

        model_predictions['distance']           += list(distance_predicted_RT)
        model_predictions['repetition']         += list(repetition_predicted_RT)
        model_predictions['error']              += list(error_predicted_RT)
        model_predictions['HCRP']               += list(HCRP_predicted_RT)
        model_predictions['low-level + HCRP']   += list(lowlevel_HCRP_predicted_RT)
        model_predictions['triplet']            += list(triplet_predicted_RT)
        model_predictions['low-level + triplet']+= list(lowlevel_triplet_predicted_RT)

        if session>12:

            model_predictions['new_triplet']            += list(new_triplet_predicted_RT)
            model_predictions['low-level + new_triplet']+= list(lowlevel_new_triplet_predicted_RT)

        else:
            model_predictions['new_triplet']            += [np.nan]*len(subdf)
            model_predictions['low-level + new_triplet']+= [np.nan]*len(subdf)

        model_predictions['train_mask']         += list(train_mask)
        model_predictions['test_mask']          += list(test_mask)
        model_predictions['index']              += list(df[(df['Subject'] == subject) & (df['Session'] == session)].index)

        ######################################################## RECENCY EFFECTS


model_predictions = pd.DataFrame(model_predictions)[[col for col in model_predictions.keys() if col not in df.columns]]
model_predictions.set_index('index', inplace=True)
data = pd.merge(df, model_predictions, left_index=True, right_index=True)
data.to_csv(cwd + '/' + 'data_and_model_predictions_forgetful.csv')
