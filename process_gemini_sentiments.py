import pandas as pd
from pathlib import Path
import numpy as np

base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

gemini_sentiments = pd.read_csv(base_dir / 'gemini_sentiments.csv')

ids = np.unique(gemini_sentiments['filename'].apply(lambda x: '_'.join(x.split('_')[:3])).astype(str))

df = pd.DataFrame(columns=['id','sentiments_gemini','probas_pos_gemini','probas_neg_gemini','probas_neu_gemini',
                           'probas_pos_norm_gemini','probas_neg_norm_gemini','probas_neu_norm_gemini'])

for id in ids:
    sentiments = []
    proba_pos = []
    proba_neg = []
    proba_neu = []

    subset = gemini_sentiments[gemini_sentiments['filename'].str.contains(id)]

    for _, row in subset.iterrows():
        sentiments.append(row['label'])
        proba_pos.append(row['POS_strength']/(row['POS_strength'] + row['NEG_strength'] + row['NEU_strength']))
        proba_neg.append(row['NEG_strength']/(row['POS_strength'] + row['NEG_strength'] + row['NEU_strength']))
        proba_neu.append(row['NEU_strength']/(row['POS_strength'] + row['NEG_strength'] + row['NEU_strength']))
    
    row = {'id': id,
           'sentiments_gemini': str(sentiments),
           'probas_pos_gemini': str(proba_pos),
           'probas_neg_gemini': str(proba_neg),
           'probas_neu_gemini': str(proba_neu),
           'norm_pos_gemini': str(np.array(proba_pos) / np.sum(proba_pos).tolist()),
           'norm_neg_gemini': str(np.array(proba_neg) / np.sum(proba_neg).tolist()),
           'norm_neu_gemini': str(np.array(proba_neu) / np.sum(proba_neu).tolist())
          }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

transcripts = pd.read_csv(Path(base_dir,'transcripts_fugu_matched_group_sentiment_phrases.csv'))
transcripts = transcripts.drop(columns=[col for col in transcripts.columns if any(x in col for x in ['_y','Fugu__'])])
transcripts.columns = ['id'] + [col.replace('_x','') + '_pysent' for col in transcripts.columns if col != 'id']

df = transcripts.merge(df, on='id', how='left')

df.to_csv(base_dir / 'transcripts_fugu_matched_group_sentiment_phrases.csv', index=False)