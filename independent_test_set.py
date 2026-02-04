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
y_labels = config["y_labels"]
tasks = config["tasks"]
bayes = config['bayes']
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

for y_label, task in itertools.product(y_labels, tasks):
    print(y_label)
    print(task)
    # Determine feature dimensions. For projects with a dictionary, pick based on the task.
    dimensions = []
    if isinstance(single_dimensions, list) and early_fusion:
        for ndim in range(1, len(single_dimensions)+1):
            for dimension in itertools.combinations(single_dimensions, ndim):
                dimensions.append("__".join(sorted(dimension)))
    else:
        dimensions = single_dimensions
        
    for dimension in dimensions:
        print(dimension)
        # Load dataset. Use CSV or Excel based on file extension.
        data_path = data_dir / (data_file_test if data_file_test.endswith(".csv") else data_file_test)
        data = pd.read_csv(data_path if data_file_test.endswith(".csv") else data_path.with_suffix(".csv"))
        
        # Identify feature columns (avoid stats and other unwanted columns)
        features = [col for col in data.columns if any(f"{x}__{y}__" in col 
                    for x,y in itertools.product(task.split("__"), dimension.split("__"))) 
                    and not isinstance(data.iloc[0][col], str)]
        # Select only the desired features along with the target and id
        data = data[features + [y_label, id_col]]
        data = data.dropna(subset=[y_label])
        
        # Separate features, target and ID.
        ID = data.pop(id_col)
        y = data.pop(y_label)

        path_to_save = Path(results_dir, task, dimension,kfold_folder,y_label,stat_folder,config['scoring_metric'] if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',version)
        
        path_to_save.mkdir(parents=True, exist_ok=True)

        data.to_csv(Path(path_to_save,'data_test.csv'),index=False)
        
        with open(str(Path(path_to_save, 'y_test.npy')), 'wb') as f:
            np.save(f,y.values)
        with open(str(Path(path_to_save, 'IDs_test.npy')), 'wb') as f:
            np.save(f,ID.values)
