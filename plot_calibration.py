import os, sys, inspect
from HCRP_LM.ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2.5)
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.sans-serif': 'Calibri'})
import matplotlib.gridspec as gridspec

cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
figdir = '\\figures_main'
figpath = cwd + figdir
if not os.path.exists(figpath):
    os.makedirs(figpath)
figpath = figpath+'\\'

posteriors = pd.read_csv(cwd + 'posterior_values_main.csv', dtype= {'subject': np.int, 'session':np.int})
MAP_data=[]
for subject in posteriors.subject.unique():
    NLL_sums = pd.pivot_table(data=posteriors[posteriors['subject']==subject], index='iteration', aggfunc=np.sum)['NLL']
    best_iter = np.argmin(NLL_sums)
    MAP_data.append(posteriors[(posteriors['subject']==subject) & (posteriors['iteration']==best_iter)])
MAP_data = pd.concat(MAP_data)
MAP_data['session'] = MAP_data['session'] - 4
MAP_data['session'] = MAP_data['session'].replace({-3:1, -2:1, -1:1, 0:1, 1:1})


df = pd.read_csv(cwd +'data_and_model_predictions_main.csv')
df = df[(df['firstACC']==1) & (df['TripC']!='XXX')]
df['TT'] = df['TT'].replace({'T':'L', 'R':'L'})

new_seq_sessions = [9,10,11,12,13,15,17]
old_seq_sessions = [14,16]

df = df[df['test_mask']==True]  # We filter the test trials before standardizing because otherwise there's a bias due to different average speed in the beginning and end of sessions

# standardized_measured_RT = []
# standardized_predicted_RT = []
# for subject in df.Subject.unique():
#     print(subject)
#     for session in df[df['Subject']==subject]['Session'].unique():
#         print(session)
#         subdf = df[(df['Subject']==subject) & (df['Session']==session)]
#         standardized_measured_RT += list(scipy.stats.zscore(subdf['measured_RT'].values))
#         standardized_predicted_RT += list(scipy.stats.zscore(subdf['low-level + HCRP'].values))
# df['measured_RT'] = standardized_measured_RT
# df['low-level + HCRP'] = standardized_predicted_RT

df = df.rename(columns={'low-level + HCRP'              : 'predicted RT',
                            'measured_RT'               : 'measured RT',
                            'TrialType'                 : 'State',
                            'TT'                        : 'P(trigram)',
                            'high_triplet_choice'       : 'P(trigram_old)',
                            'new_high_triplet_choice'   : 'P(trigram_new)'
                            })
df['P(trigram_old)'] = df['P(trigram_old)'].replace({0:'L', 1:'H'})
df['P(trigram_new)'] = df['P(trigram_new)'].replace({0:'L', 1:'H'})
df['trial type'] = (df['State'] + df['P(trigram)']).replace({'PH':'d', 'RH':'rH', 'RL':'rL'})
df['P(trigram)'] = df['P(trigram)'].replace({'T':'L','R':'L','X':'L'})
df['Session'] = df['Session'] - 4
df['Session'] = df['Session'].replace({-3:1, -2:1, -1:1, 0:1, 1:1})


session_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
session_ticks_shown = [1,2,3,4,5,6,7,8,11,15.5]
session_widths = np.array([5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1])*47
seq_type = [0]*8 + [1]*5 + [0,1,0,1]
seq_type_cmap = {0:'darkgrey', 1:'lightgrey'}

fig = plt.figure(tight_layout=True, figsize=(15,6))
gs = gridspec.GridSpec(2, 2, wspace=0.2)

data = pd.pivot_table(data=df[(df['Session']<=8)][['Subject', 'Session','measured RT','predicted RT', 'trial type']],
                        index=['Subject', 'Session', 'trial type']).reset_index()
data = data.melt(id_vars=['Subject', 'Session', 'trial type'])
data = data.rename(columns={'variable':'data'})

ax_0 = fig.add_subplot(gs[:, 0])

#dashes={'d':'-.', 'r':':'}
sns.lineplot(data=data, x='Session', y='value', hue='data', palette=['k','#E5323B'], style='trial type', dashes=False, markers={'d':'^', 'rH':'s', 'rL':'o'}, markersize=10, ax=ax_0)
handles, labels = ax_0.get_legend_handles_labels()
#handles, labels = handles[1:3]+handles[4:], labels[1:3]+labels[4:]
handles, labels = handles[4:], labels[4:]
ax_0.legend(handles=handles, labels=labels, loc='upper right', frameon=False)


data = pd.pivot_table(data=df[(df['Session']>8) & (df['State']=='R') ][['Subject', 'Session', 'measured RT','predicted RT', 'P(trigram_old)', 'P(trigram_new)']],
                        index=['Subject', 'Session', 'P(trigram_old)', 'P(trigram_new)']).reset_index()
data = data.melt(id_vars=['Subject', 'Session', 'P(trigram_old)', 'P(trigram_new)'])
data['P(trigram_old)'] = data['P(trigram_old)'].replace({'H':'rH', 'L':'rL'})
data['P(trigram_new)'] = data['P(trigram_new)'].replace({'H':'rH', 'L':'rL'})
data = data.rename(columns={'variable':'data'})

ax_1 = fig.add_subplot(gs[0, 1])

sns.lineplot(data=data, x='Session', y='value', hue='data', palette=['k','#E5323B'], dashes=False, markers={'rH':'s', 'rL':'o'}, markersize=10, legend=False, style='P(trigram_old)', ax=ax_1)
# handles, labels = ax_1.get_legend_handles_labels()
# handles, labels = handles[4:], labels[4:]
# ax_1.legend(handles=handles, labels=labels,loc='upper right', frameon=False)
ax_1.set_title('Old trigram model')

ax_2 = fig.add_subplot(gs[1, 1])

sns.lineplot(data=data, x='Session', y='value', hue='data', palette=['k','#E5323B'], dashes=False, markers={'rH':'s', 'rL':'o'}, markersize=10, legend=False, style='P(trigram_new)', ax=ax_2)
ax_2.set_title('New trigram model')

ax_0.set_xticks(session_ticks_shown[:8])
ax_1.set_xticks(session_ticks_shown[-2:])
ax_2.set_xticks(session_ticks_shown[-2:])

ax_0.set_xticklabels(range(1,9))
ax_1.set_xticklabels([])
ax_2.set_xticklabels([9, 10])

# ax_0.vlines(x=session_ticks[:8], ymin=250, ymax=255, lw=session_widths[:8], color=[seq_type_cmap[s] for s in seq_type[:8]])
# ax_1.vlines(x=session_ticks[8:], ymin=250, ymax=255, lw=session_widths[8:], color=[seq_type_cmap[s] for s in seq_type[8:]])
# ax_2.vlines(x=session_ticks[8:], ymin=250, ymax=255, lw=session_widths[8:], color=[seq_type_cmap[s] for s in seq_type[8:]])

ax_0.set_xlabel('session')
ax_1.set_xlabel('')
ax_2.set_xlabel('session')

ax_0.set_ylabel('RT [ms]')
ax_1.set_ylabel('')
ax_2.set_ylabel('')

# ax_0.set_ylim(250, 410)
# ax_1.set_ylim(250, 320)
# ax_2.set_ylim(250, 320)
#
# ax_0.set_yticks([270, 300, 330, 360, 390])
# ax_1.set_yticks([270, 300])
# ax_2.set_yticks([270, 300])

for ax in [ax_0, ax_1, ax_2]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.savefig(figpath+'calibration_plot_2.png', transparent=True)
plt.close()



#################

from statsmodels.stats.anova import AnovaRM
from sklearn.linear_model import LinearRegression

data = pd.pivot_table(data=df[df['Session']<=8][['Subject', 'Session', 'State','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'State']).reset_index()
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','State']).fit())

d = data
d['State'] = np.where(d['State']=='P', 1, 0)
d['Session*State'] = data['Session'] * data['State']
X = d[['Session', 'State', 'Session*State']]
y = d["measured RT"].values
print(LinearRegression().fit(X, y).coef_)

print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','State']).fit())

y = d["predicted RT"].values
print(LinearRegression().fit(X, y).coef_)

data = pd.pivot_table(data=df[(df['Session']<=8) & (df['State']=='R')][['Subject', 'Session', 'P(trigram)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram)']).reset_index()
data.rename(columns={'P(trigram)':'p_trigram'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram']).fit())

d = data
d['p_trigram'] = np.where(d['p_trigram']=='H', 1, 0)
d['Session*p_trigram'] = data['Session'] * data['p_trigram']
X = d[['Session', 'p_trigram', 'Session*p_trigram']]
y = d["measured RT"].values
print(LinearRegression().fit(X, y).coef_)

print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram']).fit())

y = d["predicted RT"].values
print(LinearRegression().fit(X, y).coef_)

data = pd.pivot_table(data=df[((df['Session']==8) |(df['Session']==9)) & (df['State']=='R')][['Subject', 'Session', 'P(trigram_old)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_old)']).reset_index()
data.rename(columns={'P(trigram_old)':'p_trigram_old'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram_old']).fit())
print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram_old']).fit())

data[(data['Session']==8)&(data['p_trigram_old']=='L')]['measured RT'].mean() - data[(data['Session']==8)&(data['p_trigram_old']=='H')]['measured RT'].mean()
data[(data['Session']==9)&(data['p_trigram_old']=='L')]['measured RT'].mean() - data[(data['Session']==9)&(data['p_trigram_old']=='H')]['measured RT'].mean()
data[(data['Session']==8)&(data['p_trigram_old']=='L')]['predicted RT'].mean() - data[(data['Session']==8)&(data['p_trigram_old']=='H')]['predicted RT'].mean()
data[(data['Session']==9)&(data['p_trigram_old']=='L')]['predicted RT'].mean() - data[(data['Session']==9)&(data['p_trigram_old']=='H')]['predicted RT'].mean()


################################################################################

data = df[~((df['P(trigram_old)']=='H') & (df['P(trigram_new)']=='H'))]  ## non-overlapping Hs
data = data[~((data['P(trigram_old)']=='L') & (data['P(trigram_new)']=='H'))]  ## overlapping Ls; that is, don't look at Ls that are the new Hs
data = pd.pivot_table(data=data[(data['Session']>=9)&(data['Session']<=13) & (data['State']=='R')][['Subject', 'Session', 'P(trigram_old)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_old)']).reset_index()
data.rename(columns={'P(trigram_old)':'p_trigram_old'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram_old']).fit())
print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram_old']).fit())

data = data[data['Session']==13]
data[data['p_trigram_old']=='L']['measured RT'].mean() - data[data['p_trigram_old']=='H']['measured RT'].mean()
measured_old_HvsL = data[data['p_trigram_old']=='L']['measured RT'].values - data[data['p_trigram_old']=='H']['measured RT'].values
predicted_old_HvsL = data[data['p_trigram_old']=='L']['predicted RT'].values - data[data['p_trigram_old']=='H']['predicted RT'].values


data = df[~((df['P(trigram_old)']=='H') & (df['P(trigram_new)']=='H'))]  ## non-overlapping Hs
data = data[~((data['P(trigram_old)']=='H') & (data['P(trigram_new)']=='L'))]  ## overlapping Ls; that is, don't look at Ls that used to be Hs
data = pd.pivot_table(data=data[(data['Session']>=9)&(data['Session']<=13) & (data['State']=='R')][['Subject', 'Session', 'P(trigram_new)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_new)']).reset_index()
data.rename(columns={'P(trigram_new)':'p_trigram_new'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram_new']).fit())
print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram_new']).fit())

scipy.stats.ttest_ind(data[data['p_trigram_new']=='H']['measured RT'], data[data['p_trigram_new']=='L']['measured RT'])

data = data[data['Session']==13]
data[data['p_trigram_new']=='L']['measured RT'].mean() - data[data['p_trigram_new']=='H']['measured RT'].mean()
measured_new_HvsL = data[data['p_trigram_new']=='L']['measured RT'].values - data[data['p_trigram_new']=='H']['measured RT'].values
predicted_new_HvsL = data[data['p_trigram_new']=='L']['predicted RT'].values - data[data['p_trigram_new']=='H']['predicted RT'].values


scipy.stats.ttest_rel(measured_old_HvsL, measured_new_HvsL)
plt.hist(measured_old_HvsL)
plt.hist(measured_new_HvsL)
plt.show()
scipy.stats.spearmanr(measured_old_HvsL, measured_new_HvsL)
plt.scatter(measured_old_HvsL, measured_new_HvsL)
plt.show()
scipy.stats.ttest_rel(predicted_old_HvsL, predicted_new_HvsL)
plt.hist(predicted_old_HvsL)
plt.hist(predicted_new_HvsL)
plt.show()
scipy.stats.spearmanr(predicted_old_HvsL, predicted_new_HvsL)
plt.scatter(predicted_old_HvsL, predicted_new_HvsL)
plt.show()

pearsonr(measured_old_HvsL, predicted_old_HvsL)
pearsonr(measured_new_HvsL, predicted_new_HvsL)


data = df[~((df['P(trigram_old)']=='H') & (df['P(trigram_new)']=='H'))]  ## non-overlapping Hs
data = data[~((data['P(trigram_old)']=='L') & (data['P(trigram_new)']=='H'))]  ## overlapping Ls; that is, don't look at Ls that are the new Hs
data = pd.pivot_table(data=data[((data['Session']==8) | (df['Session']==14)) & (data['State']=='R')][['Subject', 'Session', 'P(trigram_old)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_old)']).reset_index()
data.rename(columns={'P(trigram_old)':'p_trigram_old'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram_old']).fit())
print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram_old']).fit())

data[(data['Session']==8)&(data['p_trigram_old']=='L')]['measured RT'].mean() - data[(data['Session']==8)&(data['p_trigram_old']=='H')]['measured RT'].mean()
data[(data['Session']==14)&(data['p_trigram_old']=='L')]['measured RT'].mean() - data[(data['Session']==14)&(data['p_trigram_old']=='H')]['measured RT'].mean()
data[(data['Session']==8)&(data['p_trigram_old']=='L')]['predicted RT'].mean() - data[(data['Session']==8)&(data['p_trigram_old']=='H')]['predicted RT'].mean()
data[(data['Session']==14)&(data['p_trigram_old']=='L')]['predicted RT'].mean() - data[(data['Session']==14)&(data['p_trigram_old']=='H')]['predicted RT'].mean()


data = df[~((df['P(trigram_old)']=='H') & (df['P(trigram_new)']=='H'))]  ## non-overlapping Hs
data = data[~((data['P(trigram_old)']=='L') & (data['P(trigram_new)']=='H'))]  ## overlapping Ls; that is, don't look at Ls that are the new Hs
data = pd.pivot_table(data=data[(data['Session']>=14) & (data['State']=='R')][['Subject', 'Session', 'P(trigram_old)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_old)']).reset_index()
for subject in data.Subject.unique():
    for session in data.Session.unique():
        if len(data[(data['Subject']==subject)&(data['Session']==session)]) < 2:
            data = data[data['Subject']!=subject]

data.rename(columns={'P(trigram_old)':'p_trigram_old'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram_old']).fit())
print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram_old']).fit())


data = df[~((df['P(trigram_old)']=='H') & (df['P(trigram_new)']=='H'))]  ## non-overlapping Hs
data = data[~((data['P(trigram_old)']=='H') & (data['P(trigram_new)']=='L'))]  ## overlapping Ls; that is, don't look at Ls that used to be Hs
data = pd.pivot_table(data=data[(data['Session']>=14) & (data['State']=='R')][['Subject', 'Session', 'P(trigram_new)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_new)']).reset_index()
data.rename(columns={'P(trigram_new)':'p_trigram_new'}, inplace=True)
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','p_trigram_new']).fit())
print(AnovaRM(data=data, depvar='predicted RT', subject='Subject', within=['Session','p_trigram_new']).fit())



fig = plt.figure(tight_layout=True, figsize=(15,6.3))
gs = gridspec.GridSpec(2, 3, wspace=0.2)
gs.update(left=0.07,right=1)

data = pd.pivot_table(data=df[df['Session']<=8][['Subject', 'Session', 'State','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'State']).reset_index()
data = data.melt(id_vars=['Subject', 'Session', 'State'])
data['State'] = data['State'].replace({'P':'d', 'R':'r'})
data = data.rename(columns={'variable':'data'})

ax_0 = fig.add_subplot(gs[:, 0])

#dashes={'d':'-.', 'r':':'}
sns.lineplot(data=data, x='Session', y='value', hue='data', estimator=np.median, palette=['k','#E5323B'], style='State', dashes=False, markers={'d':'^', 'r':'s'}, markersize=10, ax=ax_0)
handles, labels = ax_0.get_legend_handles_labels()
#handles, labels = handles[1:3]+handles[4:], labels[1:3]+labels[4:]
handles, labels = handles[4:], labels[4:]
ax_0.legend(handles=handles, labels=['$\it{d}$'+' (deterministic)', '$\it{r}$'+' (random)'], loc='upper right', frameon=False, markerscale=2)

data = pd.pivot_table(data=df[(df['Session']<=8) & (df['State']=='R')][['Subject', 'Session', 'P(trigram)','measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram)']).reset_index()
data = data.melt(id_vars=['Subject', 'Session', 'P(trigram)'])
data = data.rename(columns={'variable':'data'})

ax_1 = fig.add_subplot(gs[:, 1])

sns.lineplot(data=data, x='Session', y='value', hue='data', estimator=np.median, palette=['k','#E5323B'], dashes=False, markers={'H':'D', 'L':'o'}, markersize=10, style='P(trigram)', ax=ax_1)
handles, labels = ax_1.get_legend_handles_labels()
handles, labels = handles[4:], labels[4:]
ax_1.legend(handles=handles, labels=['$\it{r}$'+'H (random-high)', '$\it{r}$'+'L (random-low)'], loc='upper right', frameon=False, markerscale=2)


data = df[(df['Session']>8) & (df['State']=='R')]
data = data[~((data['P(trigram_old)']=='H') & (data['P(trigram_new)']=='H'))]  ## non-overlapping Hs
data = data[~((data['P(trigram_old)']=='L') & (data['P(trigram_new)']=='H'))]  ## overlapping Ls; that is, don't look at Ls that are the new Hs
data = pd.pivot_table(data=data[['Subject', 'Session', 'P(trigram_old)', 'measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_old)']).reset_index()
data = data.melt(id_vars=['Subject', 'Session', 'P(trigram_old)'])
data = data.rename(columns={'variable':'data'})

ax_2 = fig.add_subplot(gs[0, 2])

sns.lineplot(data=data, x='Session', y='value', hue='data', estimator=np.median, palette=['k','#E5323B'], dashes=False, markers={'H':'D', 'L':'o'}, markersize=10, legend=False, style='P(trigram_old)', ax=ax_2)
# handles, labels = ax_2.get_legend_handles_labels()
# handles, labels = handles[4:], labels[4:]
# ax_2.legend(handles=handles, labels=labels, loc='upper right')
ax_2.set_title('Old trigram model')


data = df[(df['Session']>8) & (df['State']=='R')]
data = data[~((data['P(trigram_old)']=='H') & (data['P(trigram_new)']=='H'))]  ## non-overlapping
data = data[~((data['P(trigram_old)']=='H') & (data['P(trigram_new)']=='L'))]  ## overlapping Ls; that is, don't look at Ls that used to be Hs
data = pd.pivot_table(data=data[['Subject', 'Session', 'P(trigram_new)', 'measured RT','predicted RT']],
                        index=['Subject', 'Session', 'P(trigram_new)']).reset_index()
data = data.melt(id_vars=['Subject', 'Session', 'P(trigram_new)'])
data = data.rename(columns={'variable':'data'})

ax_3 = fig.add_subplot(gs[1, 2])

sns.lineplot(data=data, x='Session', y='value', hue='data', estimator=np.median, palette=['k','#E5323B'], dashes=False, markers={'H':'D', 'L':'o'}, markersize=10, legend=False, style='P(trigram_new)', ax=ax_3)
# handles, labels = ax_3.get_legend_handles_labels()
# handles, labels = handles[4:], labels[4:]
# ax_3.legend(handles=handles, labels=labels, loc='upper right')
ax_3.set_title('New trigram model')

ax_0.set_xticks(session_ticks_shown[:8])
ax_1.set_xticks(session_ticks_shown[:8])
ax_2.set_xticks(session_ticks_shown[-2:])
ax_3.set_xticks(session_ticks_shown[-2:])

ax_0.set_xticklabels(range(1,9))
ax_1.set_xticklabels(range(1,9))
ax_2.set_xticklabels([])
ax_3.set_xticklabels([9, 10])

ax_0.vlines(x=session_ticks[:8], ymin=250, ymax=255, lw=session_widths[:8], color=[seq_type_cmap[s] for s in seq_type[:8]])
ax_1.vlines(x=session_ticks[:8], ymin=250, ymax=255, lw=session_widths[:8], color=[seq_type_cmap[s] for s in seq_type[:8]])
ax_2.vlines(x=session_ticks[8:], ymin=250, ymax=255, lw=session_widths[8:], color=[seq_type_cmap[s] for s in seq_type[8:]])
ax_3.vlines(x=session_ticks[8:], ymin=250, ymax=255, lw=session_widths[8:], color=[seq_type_cmap[s] for s in seq_type[8:]])

ax_0.set_xlabel('session')
ax_1.set_xlabel('session')
ax_2.set_xlabel('')
ax_3.set_xlabel('session')

ax_0.set_ylabel('RT [ms]')
ax_1.set_ylabel('')
ax_2.set_ylabel('')
ax_3.set_ylabel('')

ax_0.set_ylim(250, 410)
ax_1.set_ylim(250, 410)
ax_2.set_ylim(250, 320)
ax_3.set_ylim(250, 320)

ax_0.set_yticks([270, 300, 330, 360, 390])
ax_1.set_yticks([270, 300, 330, 360, 390])
ax_2.set_yticks([270, 300])
ax_3.set_yticks([270, 300])

for ax in [ax_0, ax_1, ax_2, ax_3]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.savefig(figpath+'calibration_plot_new_median.png', transparent=True)
plt.close()


###################################################################################

standardized_measured_RT = []
standardized_HCRP_predicted_RT = []
standardized_triplet_predicted_RT = []
for subject in df.Subject.unique():
    print(subject)
    for session in df[df['Subject']==subject]['Session'].unique():
        print(session)
        subdf = df[(df['Subject']==subject) & (df['Session']==session)]
        standardized_measured_RT += list(scipy.stats.zscore(subdf['measured RT'].values))
        standardized_HCRP_predicted_RT += list(scipy.stats.zscore(subdf['predicted RT'].values))
        standardized_triplet_predicted_RT += list(scipy.stats.zscore(subdf['low-level + triplet'].values))
df['measured RT'] = standardized_measured_RT
df['HCRP predicted RT'] = standardized_HCRP_predicted_RT
df['trigram predicted RT'] = standardized_triplet_predicted_RT

# prevalence = {'S1':{'measured RT':[], 'predicted RT':[]}, 'S8':{'measured RT':[], 'predicted RT':[]}}
# for session in [1,8]:
#     for data in ['measured RT', 'predicted RT']:
#         for subject in df.Subject.unique():
#
#             subdf = df[(df['Subject']==subject) & (df['Session']==session)]
#             pval = scipy.stats.ttest_ind(subdf[subdf['trial type']=='d'][data], subdf[subdf['trial type']=='rH'][data])[1]
#             if pval < .05:
#                 prevalence['S'+str(session)][data].append(subject)
# true_S1_cmap        = ['salmon' if x in prevalence['S1']['measured RT'] else 'k' for x in df.Subject.unique()]
# predicted_S1_cmap   = ['red' if x in prevalence['S1']['predicted RT'] else 'k' for x in df.Subject.unique()]
# true_S8_cmap        = ['salmon' if x in prevalence['S8']['measured RT'] else 'k' for x in df.Subject.unique()]
# predicted_S8_cmap   = ['red' if x in prevalence['S8']['predicted RT'] else 'k' for x in df.Subject.unique()]

# scores = pd.DataFrame()
# for session in [1,8]:
#     for subject in df.Subject.unique():
#
#         subdf = df[(df['Subject']==subject) & (df['Session']==session)]
#
#         rH_effect = subdf[subdf['trial type']=='rH']['measured RT'].mean() - subdf[subdf['trial type']=='d']['measured RT'].mean()
#         HCRP_r2 = scipy.stats.pearsonr(subdf['measured RT'].values, subdf['predicted RT'])[0]**2
#         triplet_r2 = scipy.stats.pearsonr(subdf['measured RT'].values, subdf['low-level + triplet'])[0]**2
#         HCRP_advantage = HCRP_r2 - triplet_r2
#         scores = scores.append(pd.Series([subject, session, rH_effect, HCRP_advantage]), ignore_index=True)
#
# scores.columns = ['subject', 'session', 'rH_effect', 'HCRP_advantage']
#
# sns.regplot(data=scores[scores['session']==1], x='rH_effect', y='HCRP_advantage')
# plt.show()
# sns.regplot(data=scores[scores['session']==8], x='rH_effect', y='HCRP_advantage')
# plt.show()
# pearsonr(scores[scores['session']==1].rH_effect, scores[scores['session']==1].HCRP_advantage)
# pearsonr(scores.rH_effect, scores.HCRP_advantage)

f_1, ax_1 = plt.subplots(1, 2, figsize=(5,3.8), sharex=True, sharey=True)
f_2, ax_2 = plt.subplots(1, 2, figsize=(5,3.8), sharex=True, sharey=True)

data = pd.pivot_table(data=df[(df['Session']==1) & (df['P(trigram)']=='H')][['Subject', 'measured RT','HCRP predicted RT', 'trigram predicted RT', 'State']],
                        index=['Subject', 'State']).reset_index()
data = pd.DataFrame(data[data['State']=='R'][['measured RT','HCRP predicted RT','trigram predicted RT']].values - data[data['State']=='P'][['measured RT','HCRP predicted RT','trigram predicted RT']].values, columns=['measured', 'HCRP predicted', 'trigram predicted'])

sns.barplot(data=data.melt(), x='variable', y='value', palette=['k','#E5323B', 'steelblue'], ax=ax_1[0])
sns.regplot(data=data, x='measured', y='HCRP predicted', color='grey', ax=ax_2[0])
#ax[2].scatter(data.measured, data.predicted, c=true_S1_cmap, edgecolors=predicted_S1_cmap, linewidth=[2 if x=='red' else 1 for x in predicted_S1_cmap])

data = pd.pivot_table(data=df[(df['Session']==8) & (df['P(trigram)']=='H')][['Subject', 'measured RT','HCRP predicted RT', 'trigram predicted RT', 'State']],
                        index=['Subject', 'State']).reset_index()
data = pd.DataFrame(data[data['State']=='R'][['measured RT','HCRP predicted RT','trigram predicted RT']].values - data[data['State']=='P'][['measured RT','HCRP predicted RT', 'trigram predicted RT']].values, columns=['measured', 'HCRP predicted', 'trigram predicted RT'])

sns.barplot(data=data.melt(), x='variable', y='value', palette=['k','#E5323B','steelblue'], ax=ax_1[1])
sns.regplot(data=data, x='measured', y='HCRP predicted', color='grey', ax=ax_2[1])

for ax_i in [0,1]:
    ax_2[ax_i].spines['left'].set_color('#E5323B')
    ax_2[ax_i].spines['left'].set_linewidth(3)
    ax_2[ax_i].spines['bottom'].set_linewidth(3)
    ax_2[ax_i].spines['right'].set_visible(False)
    ax_2[ax_i].spines['top'].set_visible(False)

#ax[3].scatter(data.measured, data.predicted, c=true_S8_cmap, edgecolors=predicted_S8_cmap, linewidth=[2 if x=='red' else 1 for x in predicted_S8_cmap])

# data = pd.pivot_table(data=df[(df['Session'].isin(new_seq_sessions)) & (df['P(trigram_new)']=='H')][['Subject', 'measured RT','HCRP predicted RT', 'State']],
#                         index=['Subject', 'State']).reset_index()
# data = pd.DataFrame(data[data['State']=='R'][['measured RT','HCRP predicted RT']].values - data[data['State']=='P'][['measured RT','HCRP predicted RT']].values, columns=['measured', 'predicted'])
#
# sns.barplot(data=data.melt(), x='variable', y='value', palette=['k','#E5323B'], ax=ax[4])
# sns.regplot(data=data, x='measured', y='predicted', color='k', ax=ax[5])

ax_1[0].axhline(0, c='grey', zorder=0)
ax_1[1].axhline(0, c='grey', zorder=0)
ax_2[0].axhline(0, c='grey', zorder=0)
ax_2[0].axvline(0, c='grey', zorder=0)
ax_2[1].axhline(0, c='grey', zorder=0)
ax_2[1].axvline(0, c='grey', zorder=0)

ax_1[0].set_ylabel('higher-order effects')
ax_1[1].set_ylabel('')
ax_2[0].set_xlabel('measured')
ax_2[1].set_xlabel('measured')
# ax_2[0].set_ylabel('HCRP predicted\nhigher-order effect')
# ax_2[0].set_ylabel(r"HCRP predicted" "\n" r"$RT_{rH} - RT_{d}$")
ax_2[0].set_ylabel("HCRP predicted\n(>2)-order effect")

ax_2[1].set_ylabel('')

ax_1[0].set_xlabel('data',c='w')
ax_1[1].set_xlabel('data',c='w')
#
ax_1[0].set_ylim((-0.3,0.3))
ax_1[1].set_ylim((-0.3,0.3))
# ax_1[1].set_yticklabels([])
#
ax_2[0].set_xlim((-0.6,0.6))
ax_2[0].set_ylim((-0.6,0.6))
ax_2[1].set_xlim((-0.6,0.6))
ax_2[1].set_ylim((-0.6,0.6))
ax_2[0].set_xticks([-0.5,0.5])
ax_2[1].set_xticks([-0.5,0.5])

ax_1[0].set_xticklabels(labels=['measured','predicted'],c='w')
ax_1[1].set_xticklabels(labels=['measured','predicted'],c='w')

ax_1[0].set_title('session 1')
ax_1[1].set_title('session 8')
ax_2[0].set_title('session 1')
ax_2[1].set_title('session 8')

f_1.tight_layout()
f_2.tight_layout()

f_1.savefig(figpath+'higherorder_calibration_plot_1.png', transparent=True)
f_2.savefig(figpath+'higherorder_calibration_plot_2.png', transparent=True)
plt.close()


#################################################################################


# f, ax = plt.subplots(1, 1, figsize=(3,3.5), sharex=True, sharey=True)
#
# data = pd.pivot_table(data=df[(df['P(trigram)']=='H') & ((df['Session']==1) | (df['Session']==8))][['Subject', 'Session', 'measured RT','HCRP predicted RT', 'trigram predicted RT', 'trial type']],
#                         index=['Subject', 'Session', 'trial type']).reset_index()
# data = data.rename(columns={'Session':'session'})
# sns.pointplot(data=data.melt(['Subject','session','trial type']), x='trial type', col='session', y='value', hue='variable', scale=1.5, palette=['k','#E5323B', 'steelblue'], dodge=0.2, ax=ax)
#
# ax.get_legend().remove()
# # ax.legend(ax.get_legend_handles_labels()[0], labels=['measured', 'HCRP', 'trigram'], loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, markerscale=2)
#
# ax.axhline(0, c='grey', zorder=0)
#
# # ax.set_ylabel('higher-order effect')
# ax.set_ylabel(r'$RT_{rH} - RT_{d}$')
#
# ax.set_ylim((-0.3,0.3))
# ax.set_ylim((-0.3,0.3))
#
# f.tight_layout()
# f.savefig(figpath+'higherorder_calibration_plot_1_1.png', transparent=True)
# plt.close()


f, ax = plt.subplots(1, 1, figsize=(3,3.5), sharex=True, sharey=True)

data = pd.pivot_table(data=df[(df['P(trigram)']=='H') & ((df['Session']==1) | (df['Session']==8))][['Subject', 'Session', 'measured RT', 'HCRP predicted RT', 'trigram predicted RT', 'State']],
                        index=['Subject', 'Session', 'State']).reset_index()
print(AnovaRM(data=data, depvar='measured RT', subject='Subject', within=['Session','State']).fit())
print(AnovaRM(data=data, depvar='HCRP predicted RT', subject='Subject', within=['Session','State']).fit())
subject, session = data[data['State']=='R'].Subject.values, data[data['State']=='R'].Session.values
data = pd.DataFrame(data[data['State']=='R'][['measured RT','HCRP predicted RT','trigram predicted RT']].values - data[data['State']=='P'][['measured RT','HCRP predicted RT','trigram predicted RT']].values, columns=['measured', 'HCRP predicted RT', 'trigram predicted RT'])
data['subject'], data['session'] = subject, session

print(AnovaRM(data=data, depvar='measured', subject='subject', within=['session']).fit())
print(AnovaRM(data=data, depvar='HCRP predicted RT', subject='subject', within=['session']).fit())
print(AnovaRM(data=data, depvar='trigram predicted RT', subject='subject', within=['session']).fit())

scipy.stats.ttest_1samp(data[(data['session']==1)]['measured'], popmean=0)
scipy.stats.ttest_1samp(data[(data['session']==1)]['HCRP predicted RT'], popmean=0)
scipy.stats.ttest_1samp(data[(data['session']==8)]['measured'], popmean=0)
scipy.stats.ttest_1samp(data[(data['session']==8)]['HCRP predicted RT'], popmean=0)

pearsonr(data[(data['session']==1)]['HCRP predicted RT'], data[(data['session']==1)]['measured'])
pearsonr(data[(data['session']==8)]['HCRP predicted RT'], data[(data['session']==8)]['measured'])

pearsonr(data[(data['session']==1)]['HCRP predicted RT'], data[(data['session']==8)]['HCRP predicted RT'])

sns.pointplot(data=data.melt(['session']), x='session', y='value', hue='variable', scale=1.5, palette=['k','#E5323B', 'steelblue'], dodge=0.2, ax=ax)

ax.get_legend().remove()
# ax.legend(ax.get_legend_handles_labels()[0], labels=['measured', 'HCRP', 'trigram'], loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, markerscale=2)
ax.axhline(0, c='grey', zorder=0)
ax.set_ylim((-0.3,0.3))
ax.set_ylim((-0.3,0.3))

# ax.set_ylabel(r'$RT_{rH} - RT_{d}$')
ax.set_ylabel(r"(>2)-order effect")

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

f.tight_layout()
f.savefig(figpath+'higherorder_calibration_plot_1_1.png', transparent=True)
plt.close()
