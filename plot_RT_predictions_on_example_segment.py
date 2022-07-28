import os, sys, inspect
from HCRP_LM.ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=2.5)

cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

df = pd.read_csv(cwd+'data_and_model_predictions_session1-10_4levels_withMAP.csv')
figdir = '\\figures_main'
figpath = cwd + figdir
if not os.path.exists(figpath):
    os.makedirs(figpath)
figpath = figpath+'\\'

# subject = 104
subject = 102
#session = 3
session = 11

test_mask_generic       = [False] * 11 * 85 + [True] * 3 * 85 + [False] * 11 * 85
#test_mask_generic       = [False] * 2 * 85 + [True] * 1 * 85 + [False] * 2 * 85

df = df[(df['Subject'] == int(subject)) & (df['Session'] == int(session))][test_mask_generic][90:170][40:].reset_index()
#df = df[(df['Subject'] == int(subject)) & (df['Session'] == int(session))][test_mask_generic][40:].reset_index()

measured_RTs = df.measured_RT.values
corpus       = df.event.values
choices      = df.choice.values
TT_labels    = df.TT.replace({'X':'L', 'T':'L', 'R':'L'}).values
T = len(corpus)

intercept_component = df.intercept_component.values
intercept = intercept_component[0]
distance_component  = df.distance_component.values
repetition_component= df.repetition_component.values
error_component     = df.error_component.values
posterror_component = df.posterror_component.values
HCRP_component      = df.HCRP_component.values
lowlevel_HCRP_predicted_RTs = df['low-level + HCRP'].values

print(pearsonr(lowlevel_HCRP_predicted_RTs, measured_RTs))

seat_odds = df[['HCRP_seat_odds_'+str(i+1) for i in range(4)]].values
row_sums = seat_odds.sum(axis=1)
normalized_seat_odds = pd.DataFrame(seat_odds / row_sums[:, np.newaxis]).T.round(3)

# context_importance = df[['Session'] + ['HCRP_context_importance_'+str(i) for i in range(n_levels)]]
# KL_from_uniform = np.zeros(len(df))
# pred_distr = df[['HCRP_pred_prob_'+str(i+1) for i in range(4)]].values
# for i in range(len(pred_distr)):
#     KL_from_uniform[i] = scipy.stats.entropy(pred_distr[i], [0.25]*4)
# context_gain = np.zeros(context_importance.drop('Session', axis=1).shape)
# context_gain[:, 0] = KL_from_uniform - context_importance['HCRP_context_importance_0'].values
# for context_len in range(1,n_levels):
#     context_gain[:, context_len] = context_importance['HCRP_context_importance_' + str(context_len-1)].values - context_importance['HCRP_context_importance_' + str(context_len)].values
# context_gain = context_gain.T

pred_distr = df[['HCRP_pred_prob_'+str(i+1) for i in range(4)]].T.round(2)

model_palette = {'measured'                 : 'black',
                    'intercept'             : 'grey',
                    'spatial distance'      : '#BDD9BF',
                    'repetition'            : '#E3B578',
                    'error'                 : '#A997DF',
                    'post-error'            : '#A997DF',
                    'HCRP'                  : '#F497A2',
                    'low-level + HCRP'      : '#E5323B'}

event_palette = {1:'#D81B60', 2:'#1E88E5', 3:'#FFBF00', 4:'#004D40'}

ind = np.array(range(T)) + 0.5
#xticklabels = [str(list(TT_labels)[i]) + '\n' + str(list(corpus)[i]) + '\n' + str(list(choices)[i]) for i in range(T)]

f, ax = plt.subplots(3, 1, figsize=(13, 7.5), sharex=False, sharey=False, gridspec_kw={'height_ratios':[2,0.4,0.4]})

ax[0].bar(ind, repetition_component, color=model_palette['repetition'], label='repetition')
ax[0].bar(ind, error_component, color=model_palette['error'], label='error',\
            bottom=repetition_component)
ax[0].bar(ind, posterror_component, color=model_palette['post-error'], label='post-error',\
            bottom=[0]*len(posterror_component))
ax[0].bar(ind, distance_component, color=model_palette['spatial distance'], label='spatial distance',\
            bottom=posterror_component)
ax[0].bar(ind, HCRP_component, color=model_palette['HCRP'], label='sequence prediction\n$\mathregular{log(\it{p}_{\it{HCRP}}(\it{k_{t}}))}$',\
            bottom=posterror_component + distance_component)

ax[0].plot(ind, lowlevel_HCRP_predicted_RTs-intercept, color=model_palette['low-level + HCRP'], linewidth=6, label='predicted RT')
ax[0].plot(ind, measured_RTs-intercept, color='k', linewidth=6, label='measured RT')
ax[0].set_ylabel('RT [ms]')
ax[0].set_xticks(ind)
#ax[0].set_xticklabels(xticklabels)
#ax[0].set_xlabel('event type, event, and choice')

#ax[0].set_ylim(ax[0].get_yticks()[0]+0.05, ax[0].get_yticks()[-1:]-0.05)
#ax[0].set_yticks([-0.4, -0.2,  0. ,  0.2,  0.4,  0.6])
ax[0].axhline(y=0, c='k')
ax[0].set_xlim(0,T)
handles, labels = ax[0].get_legend_handles_labels()
label_order = ['measured RT', 'predicted RT', 'sequence prediction\n$\mathregular{log(\it{p}_{\it{HCRP}}(\it{k_{t}}))}$', 'spatial distance', 'error', 'post-error', 'repetition']
handle_order = [handles[labels.index(label)] for label in label_order]
ax[0].legend(handle_order, label_order, loc='upper left', bbox_to_anchor=(1.12, 1), frameon=False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

### I skipped the need for this by considering a segment with no filtered data (80 nonpractice trials without RT outliers)
# for i, loc in enumerate(shifted_locs):
#     if (outlier_mask[-100:][i]==False) or (practice_mask[-100:][i]==False):
#         ax[0].axvline(x=loc, c='k', alpha=0.5, lw=5)

####
ax[0].set_xticklabels([])
data_lower_lim = ax[0].get_ylim()[0]
ax[0].scatter(np.array(range(T))+0.5, [data_lower_lim -35]*T, marker='s', s=60, c=[event_palette[x] for x in corpus])
ax[0].scatter(np.array(range(T))+0.5, [data_lower_lim -50]*T, marker='s', s=60, c=[event_palette[x] for x in choices])
ax[0].set_xlabel('trial')

ax[0].annotate(s='event', xy=(T-0.05, data_lower_lim-35-5), fontsize=18)
ax[0].annotate(s='response', xy=(T-0.05, data_lower_lim-50-8), fontsize=18)

#ax[0].set_ylim(ax[0].get_ylim()[0]-10, ax[0].get_ylim()[1])
ax[0].set_yticklabels((ax[0].get_yticks()+intercept).astype('int'))

# pred_distr.index=['1', '2', '3', '4']
# sns.heatmap(pred_distr, cmap='Greys', square=True, cbar=False, annot=False, annot_kws={"size": 8}, ax=ax[1])
# ax[1].invert_yaxis()
# ax[1].set_xticklabels(['']*T)
# ax[1].set_ylabel('responses')
# ax[1].set_yticklabels(pred_distr.index, color='white')
# ax[1].set_xlabel('predictive distribution at trial')
# for _, spine in ax[1].spines.items():
#     spine.set_visible(True)

from matplotlib.colors import LinearSegmentedColormap

pred_distr.index = range(1,len(pred_distr)+1)
for i, r in pred_distr.iterrows():
    cmap = LinearSegmentedColormap.from_list('mycmap', ['white', event_palette[i]])
    for j, prob in enumerate(r):
        ax[1].vlines(x=j+0.5, ymin=i-1, ymax=i, lw=15, color=cmap(prob))

ax[1].set_xlim(0,T)
ax[1].set_ylim(0,pred_distr.shape[0])
ax[1].vlines(x=np.array(df.index[df.high_triplet==1])+0.5, ymin=0, ymax=0.3, color='k', lw=3)
ax[1].set_xticklabels(['']*T)
ax[1].set_ylabel('response')
ax[1].set_yticks(np.array(range(len(pred_distr)))+1-0.5)
ax[1].set_yticklabels(range(len(pred_distr)), c='w', rotation=90)
ax[1].set_xlabel('predictive distribution at trial')
for _, spine in ax[1].spines.items():
    spine.set_visible(True)

sns.heatmap(normalized_seat_odds, ax=ax[2], cbar=False, cmap='Greys')
ax[2].invert_yaxis()
ax[2].set_ylabel('context')  #($\it{n}$)
ax[2].set_xlabel('weight of context at trial')
ax[2].vlines(x=np.array(df.index[df.high_triplet==1])+0.5, ymin=0, ymax=0.3, color='k', lw=3)
ax[2].set_xticklabels(['']*T)
ax[2].set_yticks(np.array(range(normalized_seat_odds.shape[0]))+1-0.5)
ax[2].set_yticklabels(np.array(range(normalized_seat_odds.shape[0])), rotation=90)
for _, spine in ax[2].spines.items():
    spine.set_visible(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.45)
plt.savefig(figdir + 'trialbytrial_pred_example_late.png', transparent=True, dpi=600)
plt.close()
#
#
# f, ax = plt.subplots(2, 3, figsize=(10, 5), sharex=False, sharey=True)
#
# for ([i,j], model_component, model_label) in [([0,0], 'low-level + HCRP', 'low-level + HCRP'),\
#                                             ([0,1], 'HCRP_component', 'HCRP'),\
#                                             ([0,2], 'distance_component', 'spatial distance'),\
#                                             ([1,0], 'error_component', 'error'),\
#                                             ([1,1], 'posterror_component', 'post-error'),\
#                                             ([1,2], 'repetition_component', 'repetition'),\
#                                             ]:
#
#     if model_label == 'low-level + HCRP':
#         sns.regplot(data=df[df['test_mask']==True], x=model_component, y='measured_RT', color=model_palette[model_label], truncate=False, ax=ax[i][j])
#         ax[i][j].set_xlabel('full model')
#     else:
#         sns.regplot(data=df[df['test_mask']==True], x=model_component, y='measured_RT', color=model_palette[model_label], truncate=False, ax=ax[i][j])
#         ax[i][j].set_xlabel(model_label + ' effect')
#
#     ax[i][j].set_ylabel('')
#
#     r2 = pearsonr(df[df['test_mask']==True][model_component], df[df['test_mask']==True]['measured_RT'])[0]**2
#     ax[i][j].set_title('$r^{2}=$'+str(np.round(r2,3)), c=model_palette[model_label])
#
#     # ax[i][j].set_ylim((5.2,6))
#     # ax[i][j].set_yticks([5.3, 5.9])
#
#     ax[i][j].spines['right'].set_visible(False)
#     ax[i][j].spines['top'].set_visible(False)
#
# ax[0][0].set_ylabel('RT [ms]')
# ax[1][0].set_ylabel('RT [ms]')
# plt.tight_layout()
# plt.savefig('models_regplots_example_late.png', transparent=True, dpi=600)
# plt.close()
