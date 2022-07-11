from HCRP_LM.ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2.5)
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.sans-serif': 'Calibri'})

cwd = 'D:\\noemi_nextcloud\\Documents\\Projects\\ASRT_modeling\\human_fit_extendedlearning\\'
figdir = '\\figures_main'
figpath = cwd + figdir
if not os.path.exists(figpath):
    os.makedirs(figpath)
figpath = figpath+'\\'

# df = pd.read_csv(cwd + 'data_and_model_predictions_main.csv')  # less forgetful
df = pd.read_csv(cwd + 'data_and_model_predictions_forgetful.csv')  # more forgetful
# df = pd.read_csv(cwd + 'data_and_model_predictions_09.10.2021.csv')  # less forgetful
#df = pd.read_csv('data_and_model_predictions_01.10.2021.csv')  # more forgetful
choices = [1,2,3,4]
HCRP_choice = np.zeros(len(df))
trigram_choice = np.zeros(len(df))

for subject in df.Subject.unique():
    print(subject)

    for session in df.Session.unique():

        print(subject)
        print(session)

        subdf = df[(df['Subject']==subject) & (df['Session']==session)]
        pattern = dict(zip( subdf[subdf['TrialType']=='P']['event'][:4],
                            subdf[subdf['TrialType']=='P']['event'][1:5] ))

        for i, r in subdf.iterrows():
            probs = list(r[['HCRP_pred_prob_1', 'HCRP_pred_prob_2', 'HCRP_pred_prob_3', 'HCRP_pred_prob_4']])
            HCRP_choice[i]    = choices[probs.index(max(probs))]
            trigram_choice[i] = np.nan if math.isnan(r['event_t-2']) else pattern[r['event_t-2']]

df['HCRP_choice'], df['trigram_choice'] = HCRP_choice, trigram_choice

# df = df[df['Subject']<127]

def label_error_types(df, choice_column_name='choice'):

    error_type_code = {'pattern error'                  : 0,
                        'recency error'                 : 1,
                        'correct response repetition'   : 2,
                        'incorrect response repetition' : 3,
                        'other error'                   : 4}

    error_type = np.full(len(df), np.nan)
    error_repetition_distance = np.full(len(df), np.nan)

    for subject in df.Subject.unique():
        print(subject)

        subdf = df[(df['Subject']==subject)]
        pattern = dict(zip( subdf[subdf['TrialType']=='P']['event_t0'][:4],
                            subdf[subdf['TrialType']=='P']['event_t0'][1:5] ))

        subdf_error = subdf[(subdf['firstACC']==0)]

        for i,r in subdf_error.iloc[2:].iterrows():
            if int(r[choice_column_name]) == pattern[int(r['event_t-2'])]:
                error_type[i] = error_type_code['pattern error']

            else:

                context  = r[['event_t-2','event_t-1']].values
                response = r[choice_column_name]

                # going backwards to find first occurrence of the context
                i_ = i-1
                while (list(subdf.loc[i_][['event_t-2','event_t-1']].values) != list(context)) and (i_ > (subdf.index.min()+1)):
                    i_ -= 1

                    # only if we encountered the context can we evaluate whether it's recency error
                    if i_ > subdf.index.min():

                        if subdf.loc[i_][choice_column_name] == response:
                            if subdf.loc[i_]['firstACC'] == 1:
                                error_type[i] = error_type_code['correct response repetition']
                            else:
                                error_type[i] = error_type_code['incorrect response repetition']

                            # error_type[i] = error_type_code['recency error']

                            error_repetition_distance[i] = i - i_

                        else:
                            error_type[i] = error_type_code['other error']

    error_types_binary = pd.get_dummies(error_type)
    error_types_binary.rename(columns={v: choice_column_name + ' ' + k for k, v in error_type_code.items()}, inplace=True)
    error_types_binary.loc[np.isnan(error_type), :] = np.nan

    return error_type, error_types_binary

error_type, error_types_binary                  = label_error_types(df, 'choice')
HCRP_error_type, HCRP_error_types_binary        = label_error_types(df, 'HCRP_choice')
trigram_error_type, trigram_error_types_binary  = label_error_types(df, 'trigram_choice')

df['error_type']            = error_type
df['HCRP_error_type']       = HCRP_error_type
df['trigram_error_type']    = trigram_error_type

df = pd.concat([df, error_types_binary, HCRP_error_types_binary, trigram_error_types_binary], axis=1)

df.to_csv(cwd + 'error_data_forgetful.csv')


################################################################################


df = pd.read_csv(cwd + 'error_data_main.csv')
df = df[df['Session']<=12]  ## only analyse sessions 1-8
df_forgetful = pd.read_csv(cwd + 'error_data_01.10.2021.csv')  ## this fit was only run for sessions 1-8
df['low-level + HCRP_forgetful'] = df_forgetful['low-level + HCRP'].values


df['Session'] = df['Session'] - 4
df['Session'] = df['Session'].replace({-3:1, -2:1, -1:1, 0:1, 1:1})
df['TT'] = df['TT'].replace({'T':'L', 'R':'L'})
df = df[df['TT']!='X']

df['recency error'] = np.where((df['correct response repetition']==1) | (df['incorrect response repetition']==1), 1, 0)
df['recency error'] = np.where(np.isnan(df['correct response repetition']) | np.isnan(df['incorrect response repetition']), np.nan, df['recency error'])

# from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
#
# data = df[(df['test_mask']==True)&(df['Session']<=12)]
# data = data[(~np.isnan(data['pattern error'])) & (~np.isnan(data['HCRP_choice pattern error']))]
# sns.heatmap(confusion_matrix(data['choice'], data['HCRP_choice']), cmap='Greys',annot=True)
# plt.show()
#
# accuracy_score(data['choice'], data['HCRP_choice'], average='macro')
#
# error_types = ['pattern error', 'recency error', 'other error']
# models = ['HCRP_choice', 'trigram_choice']
#
# for error_type in error_types:
#     for model in models:
#         data = df[~math.isnan(df[error_type])]
#         confusion_matrix(data[error_type], data[model + ' ' + error_type])


#################################################################################

error_type = 'pattern error'

data = df[(df['test_mask']==True) & (df['Session']<=8) & (~np.isnan(df['error_type'])) & (df['TT']=='L')]
data = pd.pivot_table(data=data,
                        index=['Subject','Session'],
                        values=[error_type, 'HCRP_choice '+error_type, 'trigram_choice '+error_type]
                        ).reset_index().melt(id_vars=['Subject','Session'])
data['variable'] = data['variable'].replace({error_type:'measured',
                                                'trigram_choice '+error_type:'trigram',
                                                'HCRP_choice '+error_type:'HCRP'})
sns.boxplot(data=data, x='Session', y='value', hue='variable', palette = {'measured':'k',
                                                                        'HCRP':'#F497A2',
                                                                        'trigram': 'steelblue'}
                                                                        )
plt.axhline(1/3)
plt.show()

pd.pivot_table(data[data['variable']=='measured'][['Session','value']], index='Session')

error_type = 'recency error'
#error_type = 'correct response repetition'

data = df[(df['test_mask']==True) & (df['Session']<=8) & ((df[error_type]==1)|(df['other error']==1)) & (df['TT']=='L')]

data = pd.pivot_table(data=data,
                        index=['Subject','Session'],
                        values=[error_type]
                        ).reset_index()
sns.boxplot(data=data, x='Session', y=error_type)
plt.axhline(1/2)
plt.show()


df[['TT','event','choice','pattern error', 'correct response repetition', 'incorrect response repetition','recency error', 'other error']][700:750]

##################################################################################

# error_type = 'pattern error'
# error_type = 'recency error'
# data = df[(df['repetition']==0) & (df['test_mask']==True) & (df['Session']<=8) & ((df[error_type]==1)|(df['other error']==1)) & (df['TT']=='L')]
# #data = df[(df['Session']<=8) & ((df[error_type]==1)|(df['other error']==1)) & (df['TT']=='L')]
# data = pd.pivot_table(data=data,
#                         index=['Subject','Session', error_type],
#                         values=['measured_RT', 'low-level + HCRP', 'low-level + triplet']
#                         ).reset_index().melt(id_vars=['Subject','Session', error_type])
# to_drop = []
# for subject in data.Subject.unique():
#     for session in data.Session.unique():
#         subdata = data[(data['Subject']==subject) & (data['Session']==session)]
#         if len(subdata) < 6:
#             to_drop += list(subdata.index)
# data = data.drop(index=to_drop)


# barcolors=['k', 'k', '#E5323B', '#E5323B', 'steelblue', 'steelblue']
barcolors=['steelblue', 'steelblue', '#E5323B', '#E5323B', 'darkred', 'darkred', 'k', 'k']
# barcolors=['steelblue', 'steelblue', '#E5323B', '#E5323B', 'k', 'k']
f, ax = plt.subplots(1, 2, sharey=True, figsize=(9,3.7))

for i, error_type in enumerate(['pattern error', 'recency error']):

    data = df[(df['repetition']==0) & (df['test_mask']==True) & (df['Session']<=8) & ((df[error_type]==1)|(df['other error']==1)) & (df['TT']=='L')]
    data = pd.pivot_table(data=data,
                            index=['Subject','Session', error_type],
                            values=['low-level + triplet', 'low-level + HCRP', 'low-level + HCRP_forgetful', 'measured_RT']
                            # values=['low-level + triplet', 'low-level + HCRP_forgetful', 'measured_RT']
                            ).reset_index().melt(id_vars=['Subject','Session', error_type])
    to_drop = []
    subjects, sessions = [], []
    for subject in data.Subject.unique():
        for session in data.Session.unique():
            subdata = data[(data['Subject']==subject) & (data['Session']==session)]
            if len(subdata) < 6:
                to_drop += list(subdata.index)
                subjects.append(subject)
                sessions.append(session)
    data = data.drop(index=to_drop)

    error_means = []
    error_errorbars = []
    for RT in ['low-level + triplet', 'low-level + HCRP', 'low-level + HCRP_forgetful', 'measured_RT']:
    # for RT in ['low-level + triplet', 'low-level + HCRP_forgetful', 'measured_RT']:
        for error_present in [1, 0]:
            d = data[(data['variable']==RT)&(data[error_type]==error_present)].value
            error_means.append(d.mean())
            ci = scipy.stats.t.interval(0.95, len(d)-1, loc=np.mean(d), scale=scipy.stats.sem(d))
            error_errorbars.append((ci[1]-ci[0])/2)

    ax[i].bar(x         = range(len(error_means)),
                height  = error_means,
                yerr    = error_errorbars,
                color   = barcolors)

    ax[i].set_xticks(range(len(error_means)))
    ax[i].set_xticklabels([error_type.split(" ")[0], 'other']*4, rotation=90)

    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)

plt.ylim(250,320)
plt.yticks([270, 300])
ax[0].set_ylabel('error RT [ms]')
# ax[1].set_ylabel('error RT [ms]')
ax[0].set_title('pattern')
ax[1].set_title('recency')

colors = {'low-level + trigram':'steelblue', 'low-level + HCRP':'#E5323B', r'low-level + $HCRP_{f}$':'darkred', 'measured':'k'}
# colors = {'low-level + trigram':'steelblue', r'low-level + $HCRP_{f}$':'#E5323B', 'measured':'k'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
plt.savefig(figpath + 'prediction_of_error_RTs.png', dpi=600, transparent=True)
plt.close()





sns.catplot(data=data, x='variable', y='value', hue=error_type, kind='bar')
plt.ylim(250,350)
plt.show()

d = pd.pivot_table(data, index=['Subject','variable', error_type],values=['value']).reset_index()

scipy.stats.ttest_rel(d[(d['variable']=='measured_RT')&(d[error_type]==0)].value, d[(d['variable']=='measured_RT')&(d[error_type]==1)].value)
scipy.stats.ttest_rel(d[(d['variable']=='low-level + HCRP')&(d[error_type]==0)].value, d[(d['variable']=='low-level + HCRP')&(d[error_type]==1)].value)
scipy.stats.ttest_rel(d[(d['variable']=='low-level + triplet')&(d[error_type]==0)].value, d[(d['variable']=='low-level + triplet')&(d[error_type]==1)].value)

d[(d['variable']=='measured_RT')&(d[error_type]==0)].mean() -  d[(d['variable']=='measured_RT')&(d[error_type]==1)].mean()
d[(d['variable']=='low-level + HCRP')&(d[error_type]==0)].mean() - d[(d['variable']=='low-level + HCRP')&(d[error_type]==1)].mean()
d[(d['variable']=='low-level + triplet')&(d[error_type]==0)].mean() - d[(d['variable']=='low-level + triplet')&(d[error_type]==1)].mean()




data = pd.DataFrame(data[data['State']=='R'][['measured RT','predicted RT']].values - data[data['State']=='P'][['measured RT','predicted RT']].values, columns=['measured', 'predicted'])

sns.barplot(data=df[(df['firstACC']==1) | (df['other error']==1)], x='firstACC', y='distance_from_last_unigram')
plt.show()


###############################################################################

# Is the pattern error advantage significantly higher than the recency advantage?
# The slight complication for this analysis is that one subject doesn't have data for the recency errors; he has to be filtered out from the pattern error data table for the comparison

error_type = 'pattern error'
data = df[(df['repetition']==0) & (df['test_mask']==True) & (df['Session']<=8) & ((df[error_type]==1)|(df['other error']==1)) & (df['TT']=='L')]
data = pd.pivot_table(data=data,
                        index=['Subject','Session', error_type],
                        values=['measured_RT', 'low-level + HCRP', 'low-level + triplet']
                        ).reset_index().melt(id_vars=['Subject','Session', error_type])
to_drop = []
for subject in data.Subject.unique():
    for session in data.Session.unique():
        subdata = data[(data['Subject']==subject) & (data['Session']==session)]
        if len(subdata) < 6:
            to_drop += list(subdata.index)
data = data.drop(index=to_drop)
pattern_error_d = pd.pivot_table(data, index=['Subject','variable', error_type],values=['value']).reset_index()

error_type = 'recency error'
data = df[(df['repetition']==0) & (df['test_mask']==True) & (df['Session']<=8) & ((df[error_type]==1)|(df['other error']==1)) & (df['TT']=='L')]
data = pd.pivot_table(data=data,
                        index=['Subject','Session', error_type],
                        values=['measured_RT', 'low-level + HCRP', 'low-level + triplet']
                        ).reset_index().melt(id_vars=['Subject','Session', error_type])
to_drop = []
for subject in data.Subject.unique():
    for session in data.Session.unique():
        subdata = data[(data['Subject']==subject) & (data['Session']==session)]
        if len(subdata) < 6:
            to_drop += list(subdata.index)
data = data.drop(index=to_drop)
recency_error_d = pd.pivot_table(data, index=['Subject','variable', error_type],values=['value']).reset_index()

pattern_error_d = pattern_error_d[pattern_error_d['Subject'].isin(recency_error_d.Subject.unique())]

pattern_error_advantage = pattern_error_d[(pattern_error_d['variable']=='measured_RT')&(pattern_error_d['pattern error']==0)].value.values -  pattern_error_d[(pattern_error_d['variable']=='measured_RT')&(pattern_error_d['pattern error']==1)].value.values
recency_error_advantage = recency_error_d[(recency_error_d['variable']=='measured_RT')&(recency_error_d['recency error']==0)].value.values -  recency_error_d[(recency_error_d['variable']=='measured_RT')&(recency_error_d['recency error']==1)].value.values


scipy.stats.ttest_rel(pattern_error_advantage, recency_error_advantage)
