from HCRP_LM.ddHCRP_LM import *
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style="white",context='paper',font_scale=1.5)

df = pd.read_csv('data.csv')
df = df[df['TrialType']!='Prac']
df['TrialType'] = df['TrialType'].replace({'P':'d', 'R':'r'})
df['TT'] = df['TT'].replace({'X':'L', 'T':'L', 'R':'L'})
df['trial type'] = df['TrialType'] + df['TT']
df['trial type'] = df['trial type'].replace({'dH':'d'})

rh_reoccurrence_dist = df[df['trial type']=='rH']['e_to_e_distance_from_last_trigram']
p_reoccurrence_dist  = df[df['trial type']=='d']['e_to_e_distance_from_last_trigram']

f, ax = plt.subplots(1,1,figsize=(10,3.5))
ax.hist(p_reoccurrence_dist[df['e_to_e_distance_from_last_trigram']<=80], label='d', color='darkblue', alpha=0.5, bins=40)
ax.hist(rh_reoccurrence_dist[df['e_to_e_distance_from_last_trigram']<=80], label='rH', color='darkred', alpha=0.5, bins=40)
ax.axvline(p_reoccurrence_dist.median(), c='blue', lw=3)
ax.axvline(rh_reoccurrence_dist.median(), c='red', lw=3)
plt.legend()
plt.xlabel('trigram reoccurrence distance (trials)')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig('S3.png', dpi=600, transparent=True)
plt.close('all')
