import pandas as pd
from pathlib import Path

project_name = 'sj'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:/','CNC_Audio','gonza','data',project_name)

dimensions = ['pitch_analysis','talking_intervals',
              #'sentiment_analysis','verbosity','granularity','freeling','psycholinguistic_objective'
            ]
df_modified = pd.DataFrame()

for dimension in dimensions:
    df = pd.read_csv(Path(data_dir,f'{dimension}_features.csv'))

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

df_modified.to_csv(Path(data_dir,'audio_features.csv'), index=False)