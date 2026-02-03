import pandas as pd
import json,itertools,pickle
from pathlib import Path
import numpy as np

early_fusion = False

avoid_stats = ['median','std','stddev','min','max','kurtosis','skewness']

main_config = json.load(open(Path(Path(__file__).parent,'main_config.json')))
config = json.load(open(Path(Path(__file__).parent,'config.json')))
project_name = config["project_name"]

data_file_test = main_config["data_file_test"][project_name]
single_dimensions = config["single_dimensions"]
kfold_folder = config['kfold_folder']
scaler_name = config['scaler_name']
stat_folder = config['stat_folder']
tasks = config["tasks"]
bayes = config['bayes']
round_values = config['round_values']
cut_values = config['cut_values']
shuffle_labels = config['shuffle_labels']
filter_outliers = config['filter_outliers']
hyp_opt = config["n_iter"] > 0
version = config["version"]

try:
    feature_selection = config["n_iter_features"] > 0
except:
    feature_selection = config['feature_selection']
problem_type = config['problem_type']
id_col = config['id_col']

data_dir = Path(Path.home(),"data",project_name) if "/Users/gp" in str(Path.home()) else Path("D:/CNC_Audio/gonza/data",project_name)
results_dir = Path(str(data_dir).replace("data","results"))

for task in tasks:
    print(task)
    # Determine feature dimensions. For projects with a dictionary, pick based on the task.
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    
    for dimension in dimensions:

        print(dimension)
        
        y_labels = [folder.name for folder in Path(results_dir,task,dimension,kfold_folder).iterdir() if folder.is_dir()]
        
        for y_label in y_labels:
            scorings = [folder.name for folder in Path(results_dir,task,dimension,kfold_folder,y_label,stat_folder).iterdir() if folder.is_dir()]

            for scoring in scorings:
                path_to_results = Path(results_dir,task,dimension,kfold_folder,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values > 0 else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"shuffle" if shuffle_labels else "",config['version'])

                config = json.load(open(Path(path_to_results,'config.json'),'rb'))
                covars = config['covariates'] if problem_type == 'reg' else []
                regress_out = config['regress_out']

                if '_res_' in y_label:
                    print(y_label)
                # Load dataset. Use CSV or Excel based on file extension.
                data_path = data_dir / (data_file_test if data_file_test.endswith(".csv") else data_file_test)
                data = pd.read_csv(data_path if data_file_test.endswith(".csv") else data_path.with_suffix(".csv"))
                
                # Identify feature columns (avoid stats and other unwanted columns)
                features = [col for col in data.columns if any(f"{x}__{y}" in col 
                            for x,y in itertools.product(task.split("__"), dimension.split("__"))) 
                            and not isinstance(data.iloc[0][col], str)]
                # Select only the desired features along with the target and id
                data = data[features + [y_label.replace(f'_res_{config["regress_out_method"]}',''),id_col] + covars + regress_out]
                data = data.dropna(subset=[y_label.replace(f'_res_{config["regress_out_method"]}','')])
                
                # Separate features, target and ID.
                ID = data[id_col]
                y = data[y_label.replace(f'_res_{config["regress_out_method"]}','')]

                path_to_save = Path(results_dir, task, dimension,kfold_folder,y_label,stat_folder,config['scoring_metric'] if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',version)
                
                path_to_save.mkdir(parents=True, exist_ok=True)

                data.to_csv(Path(path_to_save,'data_test.csv'),index=False)
                
                with open(str(Path(path_to_save, 'y_test.npy')), 'wb') as f:
                    np.save(f,y.values)
                with open(str(Path(path_to_save, 'IDs_test.npy')), 'wb') as f:
                    np.save(f,ID.values)
