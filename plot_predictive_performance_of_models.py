import os, sys, inspect
from HCRP_LM.ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2.5)
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.sans-serif': 'Calibri'})

cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
figdir = '\\figures_main'
figpath = cwd + figdir
if not os.path.exists(figpath):
    os.makedirs(figpath)
figpath = figpath+'\\'

# seq_type = [0]*8 + [1]*5 + [0,1,0,1]
# seq_type_cmap = {0:'lightgrey', 1:'darkgrey'}
#
# session_ticks = [1,2,3,4,5,8,13,18,23,28,33,38]
# session_ticks_map = dict(zip(range(1,13), [1,2,3,4,5,8,13,18,23,28,33,38]))
# session_ticks_shown = [3,8,13,18,23,28,33,38]
# session_boundaries = [5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5]
# session_widths = np.array([1,1,1,1,1,5,5,5,5,5,5,5])*12

session_ticks = [1,2,3,4,5,8,13,18,23,28,33,38,41,42,43,44,45,46,47,48,49]
session_ticks_map = dict(zip(range(1,22), [1,2,3,4,5,8,13,18,23,28,33,38,41,42,43,44,45,46,47,48,49]))
session_ticks_shown = [3,8,13,18,23,28,33,38,43,47.5]
session_boundaries = [5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5]
session_widths = np.array([1,1,1,1,1,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1])*12

df = pd.read_csv(cwd + 'data_and_model_predictions_main.csv')
#HMM_df = pd.read_csv('data_and_HMMmodel_predictions_29.06.2021.csv')
#df = pd.merge(df, HMM_df[['HMM_component', 'low-level + HMM']], left_index=True, right_index=True)

scores = pd.DataFrame()

resp_noise = 50

model_palette = {
                    # 'distance'   : '#BDD9BF',
                    # 'repetition' : '#E3B578',
                    # 'error'      : '#A997DF',
                    'triplet'               : 'lightsteelblue',
                    'low-level + triplet'   : 'steelblue',
                    'new_triplet'            : 'lightgreen',
                    'low-level + new_triplet': 'green',
                    'HCRP'                  : '#F497A2',
                    'low-level + HCRP'      : '#E5323B',
                    #'HMM'                  : 'peachpuff',
                    #'low-level + HMM'      : 'darkorange'
                    }
model_n_params = {
                    'distance'    : 2,
                    'repetition'        : 2,
                    'error'  : 3,
                    'HCRP'              : 11,
                    'low-level + HCRP'  : 16,
                    #'HMM'              : 4+16+16+1, # TODO this is for 4 states but it should depend on the actual n_states...
                    #'low-level + HMM'  : 4+16+16+6,
                    'triplet'           : 2,
                    'low-level + triplet': 6,
                    # 'new_triplet'           : 2,
                    # 'low-level + new_triplet': 6
                    }

for subject in df.Subject.unique():
    print(subject)
    for session in df[df['Subject']==subject].Session.unique():
        print(session)
        subdf = df[(df['Subject'] == int(subject)) & (df['Session'] == int(session))]

        for evaluation, mask in [('train', 'train_mask'), ('test', 'test_mask')]:

            filt_subdf = subdf[subdf[mask]]

            for model in model_palette.keys():

                if session<13 and 'new_triplet' in model:
                    continue

                if model == 'error':
                    predicted_RT = (filt_subdf['error_component']+filt_subdf['posterror_component']).values
                elif model == 'low-level + HCRP' or model == 'low-level + HMM' or model == 'triplet' or model == 'low-level + triplet' or model == 'new_triplet' or model == 'low-level + new_triplet':
                    predicted_RT = filt_subdf[model].values
                else:
                    predicted_RT = filt_subdf[model+'_component'].values

                r2 = scipy.stats.pearsonr(filt_subdf.measured_RT.values, predicted_RT)[0]**2
                # NLL = compute_NLL(filt_subdf.measured_RT.values, predicted_RT, resp_noise, mask=[True]*len(predicted_RT))
                # AIC = compute_AIC(model_n_params[model], NLL)
                # BIC = compute_BIC(len(filt_subdf.measured_RT.values), model_n_params[model], NLL)

                #scores = scores.append(pd.Series([subject, session, model, evaluation, r2, NLL, AIC, BIC]), ignore_index=True)
                scores = scores.append(pd.Series([subject, session, model, evaluation, r2]), ignore_index=True)


#scores.columns = ['subject', 'session', 'model', 'evaluation', 'r2', 'NLL', 'AIC', 'BIC']
scores.columns = ['subject', 'session', 'model', 'evaluation', 'r2']
scores['session'] = scores['session'].astype('int')

data=scores[scores['evaluation']=='test']
data['session'] = data['session'].replace(session_ticks_map)

#f, ax = plt.subplots(1, 1, figsize=(20,6.5))
f, ax = plt.subplots(1, 1, figsize=(8, 3.5))

sns.lineplot(data=data, x='session', y='r2', hue='model',
                palette=model_palette,\
                #style='evaluation', dashes=[(2, 2),(10, 0)],
                linewidth=5, ax=ax)
handles, labels = ax.get_legend_handles_labels()
handles = [plt.Rectangle((0,0),1,1, color=handle.get_markerfacecolor()) for handle in handles]
plt.legend(handles=[handles[2], handles[3], handles[0], handles[1], handles[4], handles[5]],
            labels=['HCRP', 'low-level + HCRP', 'trigram', 'low-level + trigram', 'new trigram', 'low-level + new trigram'],
            loc='center left', bbox_to_anchor=(1, 0.5), markerscale=100, frameon=False)
# ax.get_legend().remove()
#ax.legend(loc='upper left')
ax.set_xticks(session_ticks_shown)
ax.set_xticklabels(range(1,11))
ax.set_xlim(1, np.max(session_ticks))
ymin,ymax = ax.get_ylim()
# ax.vlines(x=session_boundaries, ymin=ymin, ymax=ymax, color='k')
ax.set_ylabel("$r^{2}$")
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_ylim(0, 0.4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(figpath+'test_predictive_performance_of_HCRP_vs_triplet.png', transparent=True, dpi=600)
# plt.savefig(figpath+'test_predictive_performance.png', transparent=True, dpi=600)
plt.close()
#
# subjects = [105, 103, 124]
# f, ax = plt.subplots(1, len(subjects), figsize=(20,5))
# for i, subject in enumerate(subjects):
#     d = data[data['subject']==subject]
#     sns.lineplot(data=d, x='session', y='r2', hue='model',
#                     palette=model_palette,\
#                     #style='evaluation', dashes=[(2, 2),(10, 0)],
#                     linewidth=5, ax=ax[i])
#     ax[i].set_xticks(session_ticks_shown)
#     ax[i].set_xticklabels(range(1,11))
#     ax[i].set_xlim(1, 49)
#     #ax[i].vlines(x=session_boundaries, ymin=0, ymax=0.6, color='k')
#     ax[i].set_ylabel('')
#     ax[i].set_yticks([])
#     ax[i].set_ylim(0, 0.5)
#     ax[i].spines['right'].set_visible(False)
#     ax[i].spines['top'].set_visible(False)
#     #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax[i].get_legend().remove()
#
# ax[0].set_title('Participant ' + str(int(subjects[0])) + '\nSlower learning')
# ax[1].set_title('Participant ' + str(int(subjects[1])) + '\nFaster learning')
# ax[2].set_title('Participant ' + str(int(subjects[2])) + '\nHigher-order learning')
#
# handles, labels = ax[0].get_legend_handles_labels()
# label_order = ['low-level + HCRP', 'HCRP', 'distance', 'error', 'repetition']
# labels_to_show = ['low-level + HCRP', 'HCRP', 'spatial distance', 'error/post-error', 'repetition']
# handle_order = [handles[labels.index(label)] for label in label_order]
# #ax[0].legend(handle_order, label_order, loc='upper left', frameon=True, facecolor='black', framealpha=0.7, labelcolor='white')
# ax[2].legend(handle_order, labels_to_show, bbox_to_anchor=(1, 0.5), loc='center left', markerscale=2)
# ax[0].set_yticks([0,0.1,0.2,0.3,0.4,0.5])
# ax[0].set_ylabel("$r^{2}$")
# plt.tight_layout()
# plt.savefig(figpath + 'test_predictive_performance_across_subjects.png', transparent=True, dpi=600)
# plt.close()

# scores = scores[(scores['model']=='low-level + triplet') | (scores['model']=='low-level + HCRP')]
# scores['model'] = scores['model'].replace({'low-level + triplet':'low-level + trigram', 'triplet':'trigram'})
#
# f, ax = plt.subplots(1, 1, figsize=(7, 2.5))
# sns.barplot(data=scores[scores['evaluation']=='test'], x='session', y='r2', hue='model',
#                 palette=model_palette,\
#                 #style='evaluation', dashes=[(2, 2),(10, 0)],
#                 ax=ax)
# ax.legend(loc='upper left', labels=['unforgetful trigram', 'forgetful HCRP'], handles=ax.get_legend_handles_labels()[0], bbox_to_anchor=(1.04, 1), frameon=False)
# ax.set_ylabel("$r^{2}$")
# plt.xticks(range(8))
# plt.yticks([0, 0.25])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('test_predictive_performance_of_HCRP_vs_triplet.png', transparent=True)
# plt.close()


# f, ax = plt.subplots(1, 2, figsize=(16, 4))
# data = scores[(scores['model']=='triplet') |\
#                 (scores['model']=='HCRP')|\
#                 (scores['model']=='HMM')
#                 ][scores['evaluation']=='test']
# sns.boxplot(data=data, x='session', y='r2', hue='model',
#                 palette=model_palette,
#                 ax=ax[0])
# data = scores[(scores['model']=='low-level + triplet') |\
#                 (scores['model']=='low-level + HCRP')|\
#                 (scores['model']=='low-level + HMM')
#                 ][scores['evaluation']=='test']
# sns.boxplot(data=data, x='session', y='r2', hue='model',
#                 palette=model_palette,
#                 ax=ax[1])
# ax[0].set_ylabel("$r^{2}$")
# #ax[0].set_yticks([0, 0.25])
# ax[0].legend(loc='upper left', labels=['trigram', 'HCRP', 'HMM'], handles=ax[0].get_legend_handles_labels()[0], bbox_to_anchor=(1, 1), frameon=False)
# ax[1].legend(loc='upper left', labels=['trigram', 'HCRP', 'HMM'], handles=ax[1].get_legend_handles_labels()[0], bbox_to_anchor=(1, 1), frameon=False)
# ax[0].set_title('Internal sequence model - low-level effects')
# ax[1].set_title('Internal sequence model + low-level effects')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('test_predictive_performance_of_triplet_vs_HCRP_vs_HMM.png', transparent=True)
# plt.close()
