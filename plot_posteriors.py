import os, sys, inspect
from HCRP_LM.ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2.5)
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.sans-serif': 'Calibri'})
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
figdir = '\\figures_main'
figpath = cwd + figdir
if not os.path.exists(figpath):
    os.makedirs(figpath)
figpath = figpath+'\\'

n_levels = 4

results = pd.read_csv(cwd + 'posterior_values_main.csv', dtype= {'subject': np.int, 'session':np.int})

results.columns = [x.replace('best_','') for x in results.columns]
#results = results.drop('iteration', axis=1)
results = results[results['NLL']<10000000]
parameters = list(results.columns[4:-1])

parameter_palette={'intercept':'k',
                     'spatial_distance_coef':'#BDD9BF',
                     'repetition_coef':'#E3B578',
                     'error_coef':'#A997DF',
                     'posterror_coef':'#A997DF',
                     'prob_coef':'#F497A2'
                    }
parameter_labels ={'intercept':'intercept',
                     'spatial_distance_coef':'spatial distance',
                     'repetition_coef':'repetition',
                     'error_coef':'error',
                     'posterror_coef':'post-error',
                     #'prob_coef':'sequence prediction\nlog($\mathregular{p_{HCRP}(k_{t})}$)'
                     'prob_coef':'sequence prediction\n$\mathregular{log(\it{p}_{\it{HCRP}}(\it{k_{t}}))}$'
                    }

################### STUDY FITTED PARAMETER VALUES ###################

# strength_data = pd.pivot_table(data=results, index=['session'])[['strength_'+str(x) for x in range(1,n_levels+1)]]
# decay_data    = 1/pd.pivot_table(data=results, index=['session'])[['decayconstant_'+str(x) for x in range(1,n_levels+1)]]

# Here, the min NLL is taken for each session

# MAP_data=[]
# for subject in results.subject.unique():
#     for session in results.session.unique():
#         posterior = results[(results['subject']==subject) & (results['session']==session)]
#         MAP_data.append(posterior[posterior['NLL']==posterior['NLL'].min()])
# MAP_data = pd.concat(MAP_data)
# MAP_means = pd.pivot_table(data=MAP_data, index='session')

# Here, the iteration with the overall lowest NLL is taken; this makes sense because the sessions are interdependent via the seating arrangement

MAP_data=[]
for subject in results.subject.unique():
    NLL_sums = pd.pivot_table(data=results[results['subject']==subject], index='iteration', aggfunc=np.sum)['NLL']
    best_iter = np.argmin(NLL_sums)
    MAP_data.append(results[(results['subject']==subject) & (results['iteration']==best_iter)])
MAP_data = pd.concat(MAP_data)
MAP_means = pd.pivot_table(data=MAP_data, index='session')

##

strength_data = MAP_means[['strength_'+str(x) for x in range(1,n_levels+1)]]
decay_data = 1/MAP_means[['decayconstant_'+str(x) for x in range(1,n_levels+1)]]

df = pd.read_csv(cwd + 'data_and_model_predictions_main.csv')
# seat_odds = df[['HCRP_seat_odds_'+str(i+1) for i in range(n_levels)]].values
# row_sums = seat_odds.sum(axis=1)
# row_sums[row_sums == 0] = 0.01
# normalized_seat_odds = pd.DataFrame(seat_odds / row_sums[:, np.newaxis]).round(3)
# normalized_seat_odds['session'] = df['Session']
# normalized_seat_odds = pd.pivot_table(data=normalized_seat_odds, index='session')
# normalized_seat_odds.columns = ['seat_odds_' + str(i) for i in range(1,n_levels+1)]
# seat_odds_data = normalized_seat_odds

context_importance = df[['Session'] + ['HCRP_context_importance_'+str(i) for i in range(n_levels)]]
context_importance_data = pd.pivot_table(data=context_importance, index='Session')
context_importance_data.columns = ['context_importance_' + str(i) for i in range(n_levels)]
KL_from_uniform = np.zeros(len(df))
pred_distr = df[['HCRP_pred_prob_'+str(i+1) for i in range(4)]].values
for i in range(len(pred_distr)):
    KL_from_uniform[i] = scipy.stats.entropy(pred_distr[i], [0.25]*4)
context_gain = np.zeros(context_importance.drop('Session', axis=1).shape)
context_gain[:, 0] = KL_from_uniform - context_importance['HCRP_context_importance_0'].values
for context_len in range(1,n_levels):
    context_gain[:, context_len] = context_importance['HCRP_context_importance_' + str(context_len-1)].values - context_importance['HCRP_context_importance_' + str(context_len)].values
context_gain = pd.DataFrame(context_gain)
context_gain['Session'] = context_importance['Session']
context_gain_data = pd.pivot_table(data=context_gain, index='Session')
context_gain_data.columns = ['context_gain_' + str(i+1) for i in range(n_levels)]

f,ax=plt.subplots(1,3,figsize=(20,4))

session_ticks = [1,2,3,4,5,8,13,18,23,28,33,38,41,42,43,44,45,46,47,48,49]
session_ticks_map = dict(zip(range(1,22), [1,2,3,4,5,8,13,18,23,28,33,38,41,42,43,44,45,46,47,48,49]))
session_ticks_shown = [3,8,13,18,23,28,33,38,43,47.5]
session_boundaries = [5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5, 40.5, 45.5]
session_widths = np.array([1,1,1,1,1,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1])*7.8
seq_type = [0]*12 + [1]*5 + [0,1,0,1]
seq_type_cmap = {0:'darkgrey', 1:'lightgrey'}

norm = mpl.colors.Normalize(vmin=strength_data.min().min(), vmax=strength_data.max().max())
strength_m = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd_r)
norm = mpl.colors.Normalize(vmin=decay_data.min().min(), vmax=decay_data.max().max())
decay_m = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd_r)
norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
context_gain_m = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd)

for level in range(1,n_levels+1):
    for i, session in enumerate(strength_data.index):
        for j, data_name, data, data_cmap in [(0, 'strength_', strength_data, strength_m), (1, 'decayconstant_', decay_data, decay_m), (2, 'context_gain_', context_gain_data, context_gain_m)]:

            ax0 = ax[j].vlines(x=session_ticks[i], ymin=level-1, ymax=level, lw=session_widths[i], color=data_cmap.to_rgba(data[data_name+str(level)][session]))

            ax[j].vlines(x=session_boundaries, ymin=0, ymax=n_levels, color='k')
            ax[j].set_xlim(min(session_ticks)-0.5, max(session_ticks)+0.5)
            ax[j].set_ylim(-0.2,n_levels)
            ax[j].set_xticks(session_ticks_shown)
            ax[j].set_xticklabels(range(1,11))
            ax[j].set_yticks(np.array(range(1,n_levels+1))-0.5)
            ax[j].set_yticklabels([])
            ax[j].set_yticklabels([])
            ax[j].set_xlabel('session')
            for loc in ['bottom', 'top', 'right', 'left']:
                ax[j].spines[loc].set_visible(False)
            if level==1:
                ax[j].vlines(x=session_ticks[i], ymin=-0.2, ymax=0, lw=session_widths[i], color=seq_type_cmap[seq_type[i]])


ax_c = inset_axes(ax[0],
                   width="2%",  # width = 2% of parent_bbox width
                   height="95%",  # height : 100%
                   loc='upper left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax[0].transAxes,
                   borderpad=0,
                   )
f.colorbar(strength_m, pad=0, ticks = [math.ceil(strength_data.min().min()), math.floor(strength_data.max().max())], cax=ax_c)

ax_c = inset_axes(ax[1],
                   width="2%",  # width = 2% of parent_bbox width
                   height="95%",  # height : 100%
                   loc='upper left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax[1].transAxes,
                   borderpad=0,
                   )
# f.colorbar(decay_m, pad=0, ticks = [0.001, 0.002], cax=ax_c)
f.colorbar(decay_m, pad=0, ticks = [0.001, 0.03], cax=ax_c)

ax_c = inset_axes(ax[2],
                   width="2%",  # width = 2% of parent_bbox width
                   height="95%",  # height : 100%
                   loc='upper left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax[2].transAxes,
                   borderpad=0,
                   )
f.colorbar(context_gain_m, pad=0, ticks = [-0.1, 0.1], cax=ax_c)


ax[0].set_ylabel('context of $\it{n}$ events')
ax[0].set_yticklabels([str(i) for i in range(n_levels)])
ax[0].set_title('strength '+r'($\alpha$)', pad=10)
ax[1].set_title('forgetting rate '+r'($\lambda$)', pad=10)
ax[2].set_title('context gain ' + r'($\delta$KL)', pad=10)

plt.tight_layout(rect=[0, 0, .95, 1])
plt.subplots_adjust(wspace=0.25)
plt.savefig(figpath+'grand_average_HCRP_progression_2.png', transparent=True, dpi=400)
plt.close('all')


################################################################################

f,ax=plt.subplots(1,2,figsize=(9.5, 3))

session_ticks = [1,2,3,4,5,8,13,18,23,28,33,38]
session_ticks_map = dict(zip(range(1,12), [1,2,3,4,5,8,13,18,23,28,33,38]))
session_ticks_shown = [3,8,13,18,23,28,33,38,43,47.5]
session_boundaries = [5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5]
session_widths = np.array([1,1,1,1,1,5,5,5,5,5,5,5])*12

norm = mpl.colors.Normalize(vmin=strength_data.min().min(), vmax=strength_data.max().max())
strength_m = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
norm = mpl.colors.Normalize(vmin=decay_data.min().min(), vmax=decay_data.max().max())
decay_m = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

for level in range(1,n_levels+1):
    for i, session in enumerate(strength_data.index):
        for j, data_name, data, data_cmap in [(0, 'strength_', strength_data, strength_m), (1, 'decayconstant_', decay_data, decay_m)]:

            ax0 = ax[j].vlines(x=session_ticks[i], ymin=level-1, ymax=level, lw=session_widths[i], color=data_cmap.to_rgba(data[data_name+str(level)][session]))

            # ax[j].vlines(x=session_boundaries, ymin=0, ymax=n_levels, color='k')
            ax[j].set_xlim(min(session_ticks)-0.5, max(session_ticks)+0.5)
            ax[j].set_ylim(-0.2,n_levels)
            ax[j].set_xticks(session_ticks_shown)
            ax[j].set_xticklabels(range(1,9))
            ax[j].set_yticks(np.array(range(1,n_levels+1))-0.5)
            ax[j].set_yticklabels([])
            ax[j].set_yticklabels([])
            ax[j].set_xlabel('session')
            for loc in ['bottom', 'top', 'right', 'left']:
                ax[j].spines[loc].set_visible(False)

ax_c = inset_axes(ax[0],
                   width="3%",  # width = 2% of parent_bbox width
                   height="95%",  # height : 100%
                   loc='upper left',
                   bbox_to_anchor=(0.92, 0., 1, 1),
                   bbox_transform=ax[0].transAxes,
                   borderpad=0,
                   )
f.colorbar(strength_m, pad=0, ticks = [math.ceil(strength_data.min().min()), math.floor(strength_data.max().max())], cax=ax_c)

ax_c = inset_axes(ax[1],
                   width="3%",  # width = 2% of parent_bbox width
                   height="95%",  # height : 100%
                   loc='upper left',
                   bbox_to_anchor=(0.92, 0., 1, 1),
                   bbox_transform=ax[1].transAxes,
                   borderpad=0,
                   )
f.colorbar(decay_m, pad=0, ticks = [0.015, 0.025], cax=ax_c)

ax[0].set_ylabel('context\n($\it{n}$ events)')
ax[0].set_yticklabels([str(i) for i in range(n_levels)])
ax[0].set_title('strength '+r'($\alpha$)', pad=10)
ax[1].set_title('forgetting rate '+r'($\lambda$)', pad=10)

plt.tight_layout(rect=[0, 0, .95, 1])
plt.subplots_adjust(wspace=0.1)
plt.savefig(figpath+'grand_average_HCRP_progression.png', transparent=True, dpi=600)
plt.close('all')


################################################################################

# subjects = [105, 103, 124]
# f,ax=plt.subplots(3, len(subjects), figsize=(20,7), sharex=True, sharey=True)
# session_widths = np.array([1,1,1,1,1,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1])*7.6
#
# # seat_odds = df[['HCRP_seat_odds_'+str(i+1) for i in range(n_levels)]].values
# # row_sums = seat_odds.sum(axis=1)
# # row_sums[row_sums == 0] = 0.01
# # normalized_seat_odds = pd.DataFrame(seat_odds / row_sums[:, np.newaxis]).round(3)
# # normalized_seat_odds['session'] = df['Session']
# # normalized_seat_odds['subject'] = df['Subject']
# # normalized_seat_odds = pd.pivot_table(data=normalized_seat_odds, index=['subject','session'])
# # normalized_seat_odds.columns = ['seat_odds_' + str(i) for i in range(1,n_levels+1)]
# # seat_odds_data = normalized_seat_odds.reset_index(level=0)
#
# context_gain['Subject'] = df['Subject']
# context_gain_data = pd.pivot_table(data=context_gain, index=['Subject','Session'])
# context_gain_data.columns = ['context_gain_' + str(i+1) for i in range(n_levels)]
# context_gain_data = context_gain_data.reset_index(level=0)
#
# sub_strength_data = pd.pivot_table(data=results[results['subject'].isin(subjects)], index=['session'])[['strength_'+str(x) for x in range(1,n_levels+1)]]
# sub_decay_data = 1/pd.pivot_table(data=results[results['subject'].isin(subjects)], index=['session'])[['decayconstant_'+str(x) for x in range(1,n_levels+1)]]
# sub_context_gain_data = pd.pivot_table(data=context_gain_data[context_gain_data['Subject'].isin(subjects)], index=['Session'])[['context_gain_'+str(x) for x in range(1,n_levels+1)]]
#
# norm = mpl.colors.Normalize(vmin=sub_strength_data.min().min(), vmax=sub_strength_data.max().max())
# strength_m = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd_r)
# norm = mpl.colors.Normalize(vmin=sub_decay_data.min().min(), vmax=sub_decay_data.max().max())
# decay_m = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd_r)
# #norm = mpl.colors.Normalize(vmin=sub_context_gain_data.min().min(), vmax=sub_context_gain_data.max().max())
# norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
# context_gain_m = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd)
#
# for subj_i, subject in enumerate(subjects):
#
#     subj_strength_data = pd.pivot_table(data=results[results['subject']==subject], index=['session'])[['strength_'+str(x) for x in range(1,n_levels+1)]]
#     subj_decay_data    = 1/pd.pivot_table(data=results[results['subject']==subject], index=['session'])[['decayconstant_'+str(x) for x in range(1,n_levels+1)]]
#     subj_context_gain_data= context_gain_data[context_gain_data['Subject']==subject]
#
#     for level in range(1,n_levels+1):
#         for i, session in enumerate(strength_data.index):
#             for j, data_name, data, data_cmap in [(0, 'strength_', subj_strength_data, strength_m), (1, 'decayconstant_', subj_decay_data, decay_m), (2, 'context_gain_', subj_context_gain_data, context_gain_m)]:
#
#                 ax[j][subj_i].vlines(x=session_ticks[i], ymin=level-1, ymax=level, lw=session_widths[i], color=data_cmap.to_rgba(data[data_name+str(level)][session]))
#
#                 ax[j][subj_i].vlines(x=session_boundaries, ymin=0, ymax=n_levels, color='k')
#                 ax[j][subj_i].set_ylim(-0.2,n_levels)
#                 ax[j][subj_i].set_yticks(np.array(range(1,n_levels+1))-0.5)
#                 ax[j][subj_i].set_yticklabels([])
#                 ax[j][subj_i].set_xticklabels([])
#                 for loc in ['bottom', 'top', 'right', 'left']:
#                     ax[j][subj_i].spines[loc].set_visible(False)
#                 if level==1:
#                     ax[j][subj_i].vlines(x=session_ticks[i], ymin=-0.2, ymax=0, lw=session_widths[i], color=seq_type_cmap[seq_type[i]])
#
#     ax[0][subj_i].set_title('strength '+r'($\alpha$)', pad=10)
#     ax[1][subj_i].set_title('forgetting rate '+r'($\lambda$)', pad=10)
#     ax[2][subj_i].set_title('context gain', pad=10)
#
# ax_c = inset_axes(ax[0][len(subjects)-1],
#                    width="2%",  # width = 2% of parent_bbox width
#                    height="95%",  # height : 100%
#                    loc='upper left',
#                    bbox_to_anchor=(1.02, 0., 1, 1),
#                    bbox_transform=ax[0][len(subjects)-1].transAxes,
#                    borderpad=0,
#                    )
# f.colorbar(strength_m, pad=0, ticks = [30,70], cax=ax_c)
#
# ax_c = inset_axes(ax[1][len(subjects)-1],
#                    width="2%",  # width = 2% of parent_bbox width
#                    height="95%",  # height : 100%
#                    loc='upper left',
#                    bbox_to_anchor=(1.02, 0., 1, 1),
#                    bbox_transform=ax[1][len(subjects)-1].transAxes,
#                    borderpad=0,
#                    )
# f.colorbar(decay_m, pad=0, ticks = [0.001, 0.005], cax=ax_c)
#
# ax_c = inset_axes(ax[2][len(subjects)-1],
#                    width="2%",  # width = 2% of parent_bbox width
#                    height="95%",  # height : 100%
#                    loc='upper left',
#                    bbox_to_anchor=(1.02, 0., 1, 1),
#                    bbox_transform=ax[2][len(subjects)-1].transAxes,
#                    borderpad=0,
#                    )
# f.colorbar(context_gain_m, pad=0, ticks = [-0.1, 0.1], cax=ax_c)
#
# ax[0][0].set_ylabel('context')
# ax[1][0].set_ylabel('context')
# ax[2][0].set_ylabel('context')
#
# for subj_i in range(len(subjects)):
#     ax[2][subj_i].set_xlim(min(session_ticks)-0.5, max(session_ticks)+0.5)
#     ax[2][subj_i].set_xticks(session_ticks_shown)
#     ax[2][subj_i].set_xticklabels(range(1,11))
#     ax[2][subj_i].set_xlabel('session')
#
# ax[0][0].set_yticklabels([str(i) for i in range(n_levels)])
# ax[1][0].set_yticklabels([str(i) for i in range(n_levels)])
# ax[2][0].set_yticklabels([str(i) for i in range(n_levels)])
#
# #plt.tight_layout(rect=[0, 0, .95, 1])
# plt.subplots_adjust(wspace=0.1, hspace=0.4)
# plt.savefig(figpath+'grand_average_HCRP_progression_across_subjects_2.png', transparent=True, dpi=600)
# plt.close('all')

#

#d = pd.pivot_table(data=MAP_data, index=['session','subject'])[parameters[-6:]].reset_index().melt(id_vars=['session','subject'])
d = pd.pivot_table(data=results, index=['session','subject'])[parameters[-6:]].reset_index().melt(id_vars=['session','subject'])
d['session']=d['session'].astype('int')
f,ax=plt.subplots(1,6, figsize=(20,4))
for i, parameter in enumerate(d.variable.unique()):
    data = d[d['variable']==parameter]
    data['session'] = data['session'].replace(session_ticks_map)
    sns.lineplot(data=data, x='session', y='value', err_style="bars", color=parameter_palette[parameter], linewidth = 3, ax=ax[i])
    if i==0:
        #ax[i].set_ylabel('coefficient\n(log('+ r'$\tau$'+'))')
        ax[i].set_ylabel('coefficient ('+ r'$\tau$'+')')
    else:
        ax[i].set_ylabel('')

    ax[i].set_title(parameter_labels[parameter])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)

    #ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax[i].set_xticks(session_ticks_shown)
    ax[i].set_xticklabels(range(1,11))
    ymin,ymax = ax[i].get_ylim()
    proportion_range = (ymax-ymin) * 0.05
    ax[i].vlines(x=session_boundaries, ymin=ymin, ymax=ymax, color='k')
    if ymin<0 and ymax>0:
        ax[i].hlines(y=0, xmin=ax[i].get_xlim()[0], xmax=ax[i].get_xlim()[1], color='k')

    for j in range(len(session_ticks)):
        ax[i].vlines(x=session_ticks[j]+0.5, ymin=ymin, ymax=ymin+proportion_range, lw=session_widths[j]*0.45, color=seq_type_cmap[seq_type[j]])

plt.tight_layout()
f.subplots_adjust(wspace=0.25)
plt.savefig(figpath+'grand_average_param_progression_lowlevel.png', transparent=True, dpi=400)
plt.close('all')

###########################    ANALYSIS   #################################

responsevar_data = pd.pivot_table(data=results, index=['session','subject'])[parameters[-6:]].reset_index()

strength_data = pd.pivot_table(data=results, index=['session', 'subject'])[['strength_'+str(x) for x in range(1,n_levels+1)]].reset_index().drop(['subject','session'], axis=1)
decay_data    = 1/pd.pivot_table(data=results, index=['session', 'subject'])[['decayconstant_'+str(x) for x in range(1,n_levels+1)]].reset_index().drop(['subject','session'], axis=1)

seat_odds = df[['HCRP_seat_odds_'+str(i+1) for i in range(n_levels)]].values
row_sums = seat_odds.sum(axis=1)
row_sums[row_sums == 0] = 0.01
normalized_seat_odds = pd.DataFrame(seat_odds / row_sums[:, np.newaxis]).round(3)
normalized_seat_odds['session'] = df['Session']
normalized_seat_odds['subject'] = df['Subject']
normalized_seat_odds = pd.pivot_table(data=normalized_seat_odds, index=['session','subject'])
normalized_seat_odds.columns = ['seat_odds_' + str(i) for i in range(1,n_levels+1)]
seat_data = normalized_seat_odds.reset_index().drop(['subject','session'], axis=1)
dfs = [responsevar_data, strength_data, decay_data, seat_data]

from functools import reduce
data = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True), dfs)

import statsmodels.api as sm
import statsmodels.formula.api as smf

for variable in data.columns[2:]:
    ## Dynamic change in initial 5 epochs:
    #d = data
    #d = data[data['session']<=5]
    d = data[(data['session']>5) & (data['session']<=12)]
    #d = data[(data['session']==12) | (data['session']==13)]
    #d = data[(data['session']>=13) & (data['session']<18)]
    #d = data[(data['session']>12)]

    md = smf.mixedlm(variable + " ~ session", d, groups=d['subject'])
    mdf = md.fit()
    #print(mdf.summary())
    print('---')
    print(variable)
    print(mdf.pvalues['session'])

HL_data = pd.pivot_table(data=df[df['train_mask']], index=['Subject', 'Session', 'high_triplet'], values=['measured_RT']).reset_index()
score = HL_data[HL_data['high_triplet']==0].measured_RT.values - HL_data[HL_data['high_triplet']==1].measured_RT.values
HL_data = HL_data[HL_data['high_triplet']==0][['Subject','Session']]
HL_data['HL_score'] = score
HL_data.sort_values(['Session','Subject'], inplace=True)
data['HL_score'] = HL_data['HL_score'].values

for variable in data.columns[2:-1]:
    ## Dynamic change in initial 5 epochs:
    # d = data
    d = data[data['session']<=5]
    #d = data[(data['session']>5) & (data['session']<=12)]
    # d = data[(data['session']==12) | (data['session']==13)]
    #d = data[(data['session']>=13) & (data['session']<18)]
    # d = data[(data['session']>12)]

    md = smf.mixedlm('HL_score ~ session + ' + variable, d, groups=d['subject'])
    mdf = md.fit()
    #print(mdf.summary())
    print('---')
    print(variable)
    print(mdf.pvalues[variable])


for variable in data.columns[2:-1]:
    ## Dynamic change in initial 5 epochs:
    # d = data
    # d = data[data['session']<=5]
    # d = data[(data['session']>5) & (data['session']<=12)]
    #d = data[(data['session']==12) | (data['session']==13)]
    d = data[(data['session']==16)  | (data['session']==17)]
    # d = data[(data['session']>=13) & (data['session']<18)]
    # d = data[(data['session']>12)]
    d = data[(data['session']>=18)]

    d = pd.pivot_table(data=d, index='subject', values=['HL_score', variable]).reset_index()

    if pearsonr(d[variable], d['HL_score'])[1]<.05:

        plt.scatter(d[variable], d['HL_score'])
        plt.title(variable)
        plt.show()
