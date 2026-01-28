import pickle 
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis, skew
import itertools

project_name = 'affective_pitch'

base_dir =  Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name) 

stats = ['mean','std','min','max','kurt','skew']
with open(Path(base_dir,'embeddings.pkl'),'rb') as f:
    embeddings = pickle.load(f)

embeddings['id'] = embeddings['audio_path']

ids = embeddings['audio_path']

windows = np.unique(ids.apply(lambda x: x.split('__')[1].split('.')[0]))

mean_embeddings_df = pd.DataFrame(columns=['id'] + [f'Fugu__embeddings__{window}_{stat}' for window,stat in itertools.product(windows,stats)])

for id in np.unique(ids):
    embeddings_mean = embeddings.loc[embeddings.id.isin([id]),'z'].mean()[0]    
    window = id.split('__')[1].split('.')[0]
    stats_ = {'id': id.split('_')[0],
              f'Fugu__embeddings__{window}_mean':np.nanmean(embeddings_mean),
              f'Fugu__embeddings__{window}_std':np.nanstd(embeddings_mean),
              f'Fugu__embeddings__{window}_min':np.nanmin(embeddings_mean),
              f'Fugu__embeddings__{window}_max':np.nanmax(embeddings_mean),
              f'Fugu__embeddings__{window}_kurt':kurtosis(embeddings_mean),
              f'Fugu__embeddings__{window}_skew':skew(embeddings_mean)}
    if id.split('_')[0] in mean_embeddings_df['id'].values:
        stats_.pop('id')
        index = mean_embeddings_df.index[mean_embeddings_df['id'] == id.split('_')[0]][0]
    else:
        index = len(mean_embeddings_df)
    mean_embeddings_df.loc[index,list(stats_.keys())] = list(stats_.values())

labels = pd.read_csv(Path(base_dir,'matched_ids.csv'))[['id','group','sex','age','site']]
mean_embeddings_df = labels.merge(mean_embeddings_df,how='inner')

mean_embeddings_df.to_csv(Path(base_dir,'embeddings_df_windows.csv'),index=False)
