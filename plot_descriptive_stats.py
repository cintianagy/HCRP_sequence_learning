import os, sys, inspect
from ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2)
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.sans-serif': 'Calibri'})

cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
figdir = '\\figures_main'
df = pd.read_csv(cwd + '/' + 'data_and_model_predictions_forgetful.csv')
df = df[df['TT']!='X']
df['TT'] = df['TT'].replace({'T':'L', 'R':'L'})

standardized_measured_RT = []
standardized_predicted_RT = []
for subject in df.Subject.unique():
    print(subject)
    for session in df[df['Subject']==subject]['Session'].unique():
        print(session)
        subdf = df[(df['Subject']==subject) & (df['Session']==session)]
        standardized_measured_RT += list(scipy.stats.zscore(subdf['measured_RT'].values))
        standardized_predicted_RT += list(scipy.stats.zscore(subdf['low-level + HCRP'].values))
df['measured_RT']       = standardized_measured_RT
df['low-level + HCRP']  = standardized_predicted_RT


# f, ax = plt.subplots(1, 2, figsize=(3.72, 1.7), sharey=True, sharex=False)
# data = df[(df['Session']==12)].pivot_table(index=['Subject', 'TrialType']).reset_index()
# ax[0].bar(['$\it{d}$', '$\it{r}$'], [data[data['TrialType']=='P'].measured_RT.mean(),
#                                     data[data['TrialType']=='R'].measured_RT.mean()],
#                                     yerr=[data[data['TrialType']=='P'].measured_RT.sem(),
#                                         data[data['TrialType']=='R'].measured_RT.sem()],
#             color='Grey'
#             )
#
# data = df[(df['TrialType']=='R') & (df['Session']==12)].pivot_table(index=['Subject', 'TT']).reset_index()
# ax[1].bar(['$\it{r}$'+'H', '$\it{r}$'+'L'], [data[data['TT']=='H'].measured_RT.mean(),
#                         data[data['TT']=='L'].measured_RT.mean()],
#                         yerr=[data[data['TT']=='H'].measured_RT.sem(),
#                                 data[data['TT']=='L'].measured_RT.sem()],
#             color='Grey'
#             )
# plt.ylim(-0.25, 0.25)
# # ax[0].set_yticks([5.3, 5.9])
# ax[0].set_ylabel('std. RT')
# for i in [0,1]:
#     ax[i].spines['right'].set_visible(False)
#     ax[i].spines['top'].set_visible(False)
#     ax[i].spines['bottom'].set_visible(False)
#     ax[i].axhline(0, c='k')
# plt.tight_layout()
# plt.savefig('descriptive_stats.png', dpi=1500, transparent=True)
# plt.close()

f, ax = plt.subplots(1, 2, figsize=(3.72, 1.7), sharey=True, sharex=False)

data = df[(df['Session']==12)].pivot_table(index=['Subject', 'TrialType'], values='measured_RT').reset_index()
sns.pointplot(data=data, y='measured_RT', x='TrialType', color='k', ax=ax[0])
ax[0].set_xticklabels(['$\it{d}$', '$\it{r}$'])

data = df[(df['TrialType']=='R') & (df['Session']==12)].pivot_table(index=['Subject', 'TT'], values='measured_RT').reset_index()
sns.pointplot(data=data, y='measured_RT', x='TT', color='k', ax=ax[1])
ax[1].set_xticklabels(['$\it{r}$'+'H', '$\it{r}$'+'L'])
plt.ylim(-0.4, 0.4)
# ax[0].set_yticks([5.3, 5.9])
ax[0].set_ylabel('std. RT')
ax[1].set_ylabel('')
ax[0].set_xlabel('')
ax[1].set_xlabel('')
for i in [0,1]:
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].axhline(0, c='k')
plt.tight_layout()
plt.savefig(cwd + 'descriptive_stats.png', dpi=1500, transparent=True)
plt.close()

################################################################################

f, ax = plt.subplots(1, 6, figsize=(10, 1.7), sharey=True, sharex=False)
data = df[(df['Session']==12)].pivot_table(index=['Subject', 'TrialType']).reset_index()
ax[0].bar(['$\it{d}$', '$\it{r}$'], [data[data['TrialType']=='P'].measured_RT.mean(),
                                    data[data['TrialType']=='R'].measured_RT.mean()],
                                    yerr=[data[data['TrialType']=='P'].measured_RT.sem(),
                                        data[data['TrialType']=='R'].measured_RT.sem()],
            color='Grey'
            )
ax[0].set_title('state')

data = df[(df['TrialType']=='R') & (df['Session']==12)].pivot_table(index=['Subject', 'TT']).reset_index()
ax[1].bar(['$\it{r}$'+'H', '$\it{r}$'+'L'], [data[data['TT']=='H'].measured_RT.mean(),
                        data[data['TT']=='L'].measured_RT.mean()],
                        yerr=[data[data['TT']=='H'].measured_RT.sem(),
                                data[data['TT']=='L'].measured_RT.sem()],
            color='Grey'
            )
ax[1].set_title('p(trigram)')

data = df[(df['repetition_x']==0) & (df['Session']==12)].pivot_table(index=['Subject', 'spatial_distance']).reset_index()
ax[2].bar(['$\it{1}$', '$\it{2}$', '$\it{3}$'], [data[data['spatial_distance']==1].measured_RT.mean(),
                                                data[data['spatial_distance']==2].measured_RT.mean(),
                                                data[data['spatial_distance']==3].measured_RT.mean()],
                                                yerr=[data[data['spatial_distance']==1].measured_RT.sem(),
                                                        data[data['spatial_distance']==2].measured_RT.sem(),
                                                        data[data['spatial_distance']==3].measured_RT.sem()],
            color='Grey'
            )
ax[2].set_title('spatial distance')

data = df[(df['Session']==12)].pivot_table(index=['Subject', 'repetition_x']).reset_index()
ax[3].bar(['$\it{yes}$', '$\it{no}$'], [data[data['repetition_x']==1].measured_RT.mean(),
                        data[data['repetition_x']==0].measured_RT.mean()],
                        yerr=[data[data['repetition_x']==1].measured_RT.sem(),
                                data[data['repetition_x']==0].measured_RT.sem()],
            color='Grey'
            )

ax[3].set_title('repetition')

data = df[(df['Session']==12)].pivot_table(index=['Subject', 'error_x']).reset_index()
ax[4].bar(['$\it{yes}$', '$\it{no}$'], [data[data['error_x']==1].measured_RT.mean(),
                                                data[data['error_x']==0].measured_RT.mean()],
                                                yerr=[data[data['error_x']==1].measured_RT.sem(),
                                                        data[data['error_x']==0].measured_RT.sem()],
            color='Grey'
            )
ax[4].set_title('error')

data = df[(df['Session']==12)].pivot_table(index=['Subject', 'posterror']).reset_index()
ax[5].bar(['$\it{yes}$', '$\it{no}$'], [data[data['posterror']==1].measured_RT.mean(),
                                                        data[data['posterror']==0].measured_RT.mean()],
                                                        yerr=[data[data['posterror']==1].measured_RT.sem(),
                                                        data[data['posterror']==0].measured_RT.sem()],
            color='Grey'
            )
ax[5].set_title('post-error')

ax[0].set_ylabel('std. RT')
for i in range(6):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].axhline(0, c='k')
plt.tight_layout()
plt.savefig('high_and_lowlevel_descriptive_stats.png', dpi=1500, transparent=True)
plt.close()

################################################################################

f, ax = plt.subplots(1, 2, figsize=(3.72, 1.7), sharey=True, sharex=False)
data = pd.DataFrame([['d',1],['r', 0.25]], columns=['TrialType', 'probability'])
sns.pointplot(data=data, y='probability', x='TrialType', color='k', ax=ax[0])
ax[0].set_xticklabels(['$\it{d}$', '$\it{r}$'])

data = pd.DataFrame([['rh',0.25],['rl', 0.25]], columns=['TT', 'probability'])
sns.pointplot(data=data, y='probability', x='TT', color='k', ax=ax[1])
ax[1].set_xticklabels(['$\it{r}$'+'H', '$\it{r}$'+'L'])
plt.ylim(0, 1.1)
ax[0].invert_yaxis()
ax[0].set_ylabel('ground\ntruth\nprobability')
ax[1].set_ylabel('')
ax[0].set_xlabel('')
ax[1].set_xlabel('')
for i in [0,1]:
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].axhline(0, c='k')
    ax[i].axhline(0.25, c='k', linestyle='--')
plt.tight_layout()
plt.savefig('descriptive_stat_prediction_under_truemodel.png', dpi=1500, transparent=True)
plt.close()

f, ax = plt.subplots(1, 2, figsize=(3.72, 1.7), sharey=True, sharex=False)
data = pd.DataFrame([['d',0.625],['r', 0.125]], columns=['TrialType', 'probability'])
sns.pointplot(data=data, y='probability', x='TrialType', color='k', ax=ax[0])
ax[0].set_xticklabels(['$\it{d}$', '$\it{r}$'])

data = pd.DataFrame([['rh',0.625],['rl', 0.125]], columns=['TT', 'probability'])
sns.pointplot(data=data, y='probability', x='TT', color='k', ax=ax[1])
ax[1].set_xticklabels(['$\it{r}$'+'H', '$\it{r}$'+'L'])
plt.ylim(0, 1)
ax[0].invert_yaxis()
ax[0].set_ylabel('probability')
ax[1].set_ylabel('')
ax[0].set_xlabel('')
ax[1].set_xlabel('')
for i in [0,1]:
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].axhline(0, c='k')
    ax[i].axhline(0.25, c='k', linestyle='--')
plt.tight_layout()
plt.savefig('descriptive_stat_under_trigrammodel.png', dpi=1500, transparent=True)
plt.close()

################################################################################
