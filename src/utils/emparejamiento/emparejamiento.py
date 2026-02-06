import pandas as pd
import numpy as np
from tableone import TableOne
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from matching_module import perform_matching

project_name = 'crossling_mci'
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

target_vars = ['group']

matching_vars = ['sex','age']
    
fact_vars = ['sex']
cont_vars = ['age']

for target_var in target_vars:
    filename = f'all_data_{target_var}.csv'

    print(target_var)
    # Define variables
    vars = matching_vars + [target_var,'id']

    output_var = target_var

    data = pd.read_csv(Path(data_dir,filename))

    data.dropna(subset=[target_var],inplace=True)
    for fact_var in fact_vars:
        data[fact_var] = data[fact_var].astype('category').cat.codes

    caliper = 0.05

    table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
    print(table_before)

    matched_data = perform_matching(data, output_var,matching_vars,fact_vars,caliper=caliper)

    matched_data = matched_data.drop_duplicates(subset='id')

    # Save tables and matched data
    #print(table_before)

    table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
    print(table)

    matched_data.to_csv(Path(data_dir,f'{filename.split(".")[0]}_matched_{target_var}.csv'.replace('__','_')), index=False)
    table_before.to_csv(Path(data_dir,f'table_before_{filename.split(".")[0]}_{target_var}.csv'.replace('__','_')))
    table.to_csv(Path(data_dir,f'table_matched_{filename.split(".")[0]}_{target_var}.csv'.replace('__','_')))