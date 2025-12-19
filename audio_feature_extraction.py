from pathlib import Path
import sys,tqdm,itertools,json
import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore')

subsets = ['audio']

dimensions = {
              'audio':['talking_intervals','pitch_analysis']
        }

file_extentions = {'audio':'.wav','nlp':'.txt'}

project_name = 'sj'

for subset in subsets:
    for dimension in dimensions[subset]:
        sys.path.append(str(Path(Path.home(),'tell','local_feature_extraction',f'{subset}_features',dimension)))

        from app import main

        data_dir = Path(Path.home(),'data',project_name,'audios_prepro')

        extracted_features = pd.DataFrame(columns=['id'])
        files = [file for file in data_dir.iterdir() if file.suffix == file_extentions[subset]]

        for file in tqdm.tqdm(files):
            filename = file.stem.replace('Preg','Pre').replace('pre','Pre').replace('post','Post').replace('2Pos1','2Post1').replace('2Pos2','2Post2')
            if not Path(data_dir.parent,'transcripciones',f'{filename}_mono_16khz_diarize_loudnorm_denoised.txt').exists():
                continue
            try:
                features_dict = main(str(file))
                if 'data' in features_dict.keys():
                    features = features_dict['data'].copy()
                else:
                    features_dict = json.loads(features_dict['body']).copy()
                    features = features_dict['data'].copy()
                features = {f'{dimension}__{k}': v for k, v in features.items() if not isinstance(v, list)}

            except:
                print(f'Error extracting features from {file.stem}')
                continue
            #Drop items from features_dict['scores'] that are lists
            features_dict['scores'] = {k: v for k, v in features_dict['scores'].items() if not isinstance(v, list)}
            features.update(features_dict['scores'])

            features['id'] = '_'.join(file.stem.split('_')[:2])
            features['task'] = file.stem.split('_')[2]

            if extracted_features.empty:
                extracted_features = pd.DataFrame(features, index=[0])
            else:
                extracted_features.loc[len(extracted_features)] = features
            
        #Remove app from sys.path
        sys.path.remove(str(Path(Path.home(),'tell','local_feature_extraction',f'{subset}_features',dimension)))

        extracted_features['task'] = extracted_features['task'].apply(lambda x: x.replace('Preg','Pre').replace('pre','Pre').replace('post','Post').replace('2Pos1','2Post1').replace('2Pos2','2Post2'))

        extracted_features.to_csv(Path(Path.home(),'data',project_name,f'{dimension}_features.csv'),index=False)