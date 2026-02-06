from pathlib import Path
import sys,tqdm

sys.path.append(str(Path(Path.home(),'tell','local_feature_extraction','audio_features','preprocessing')))

project_name = 'sj'
file_ext = '.mp3'

from audio_preprocess_pipeline import preprocess_audio

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

path_to_audios = Path(data_dir,'AUDIOS PROCESADOS')
for audio in tqdm.tqdm(path_to_audios.rglob('*'+file_ext)):
    print(f'Preprocessing {audio.stem}')
    filename = audio.stem.replace({'Preg':'Pre','pos':'Pos','pre':'Pre'})

    if Path(path_to_audios.parent,'diarize',f'{filename}_mono_16khz_loudnorm_denoised.wav').exists():
        continue

    preprocess_audio(audio,str(path_to_audios),
                    diarize_config=False,
                    vad_config=True)