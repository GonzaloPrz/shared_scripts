import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from matching_module import *

project_name = 'affective_pitch'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

target_vars = ['group']

filenames = [
             'filtered_data.csv'
            ]

for filename in filenames:
    try:
        task = filename.split('__')[2]
    except:
        task = ''
    for target_var in target_vars:
        print(target_var)
        # Define variables
        vars = ['sex','age','education','site',target_var, 'id']
        output_var = target_var
        
        matching_vars = ['age','sex','site']

        fact_vars = ['sex','site']
        cont_vars = ['age']

        data = pd.read_csv(Path(data_dir,filename))

        data.dropna(subset=[target_var] ,inplace=True)

        data[target_var] = data[target_var]
        for fact_var in fact_vars:
            data[fact_var] = data   [fact_var].astype('category').cat.codes

        for var in matching_vars:
            data.dropna(subset=var,inplace=True)

        caliper = 0.5

        matched_data = perform_three_way_matching(data, output_var,matching_vars,fact_vars,treatment_values=('AD','FTD','CN'),caliper=caliper)
        matched_data = matched_data.drop_duplicates(subset='id')

        # Save tables and matched data
        table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])

        table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
        print(table_before)
        print(table)

        matched_data.to_csv(Path(data_dir,f'{filename.split(".")[0]}_matched_{target_var}.csv'.replace('__','_')), index=False)
        table_before.to_csv(Path(data_dir,f'table_before_{filename.split(".")[0]}_{target_var}.csv'.replace('__','_')))
        table.to_csv(Path(data_dir,f'table_matched_{filename.split(".")[0]}_{target_var}.csv'.replace('__','_')))