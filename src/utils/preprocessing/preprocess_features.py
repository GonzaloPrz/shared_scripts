import pandas as pd
from pathlib import Path
import numpy as np

project_name = 'continuo_ivo'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:/','CNC_Audio','gonza','data',project_name)

dimensions = [file.name for file in data_dir.iterdir() if file.is_file and file.name.endswith('_features.csv')]

df_modified = pd.DataFrame()

for dimension in dimensions:
    df = pd.read_csv(Path(data_dir,dimension))

    if 'task' in df.columns:
        tasks = df['task'].unique()

        for task in tasks:
            df_task = df[df['task'] == task]
            for col in df_task.columns:
                if col == 'id':
                    continue
                df_task[f'{task}__{col}'] = df_task[col]
                df_task.drop(columns=[col], inplace=True)

            if df_modified.empty:
                df_modified = df_task.reset_index(drop=True)
            else:
                df_modified = pd.merge(df_modified, df_task, on='id',how='outer')
    else:
        if df_modified.empty:
            df_modified = df
        else:
            df_modified = pd.merge(df_modified,df,on='id',how='outer')

feature_names = np.unique(['__'.join(col.split('__')[1:3]) for col in df_modified.columns if any(x in col for x in ['letra','animales']) and all(x not in col for x in ['list','data','query','error','text','nouns','verbs','segments'])])

for feature_name in feature_names:
    df_modified[f'fas__{feature_name}'] = np.nanmean(df_modified[[f'letra_f__{feature_name}',
                                                       f'letra_a__{feature_name}',
                                                       f'letra_s__{feature_name}']],axis=1)
    
    df_modified[f'mean__{feature_name}'] = np.nanmean(df_modified[[f'letra_f__{feature_name}',
                                                      f'letra_a__{feature_name}',
                                                      f'letra_s__{feature_name}',
                                                      f'animales__{feature_name}']],axis=1)

df_modified = df_modified.drop([col for col in df_modified.columns if any(x in col for x in ['list','data','query','error','text','nouns','verbs','segments'])],axis=1)

labels = pd.read_csv(Path(data_dir,'all_labels.csv'))
df_modified = pd.merge(df_modified,labels,on='id',how='outer')

df_modified.to_csv(Path(data_dir,'all_data.csv'), index=False)