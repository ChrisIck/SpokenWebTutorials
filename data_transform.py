import numpy as np
import pandas as pd
import os
import ffmpeg
from IPython.display import Audio

cv_dir = "/scratch/ci411/spokenweb/commonvoice/"
cvout_dir = "/scratch/ci411/spokenweb/commonvoice_reshaped/"
train_df = pd.read_csv(cv_dir + 'train.tsv', sep='\t')

min_utter = 10
train_grouped = train_df.groupby('client_id', sort=False)
train_grouped_df = train_grouped.count().reset_index()
train_sorted = train_grouped_df.sort_values('path', ascending=False).reset_index()
train_sorted_fil = train_sorted[train_grouped_df['path']>min_utter]
print('All samples:{}'.format(train_sorted.shape[0]))
print('Filtered Samples with at least {} utterances: {}'.format(min_utter, train_sorted_fil.shape[0]))

for speaker in train_sorted_fil['client_id']:
    speaker_dir = os.path.join(cvout_dir, speaker)
    audio_dir = os.path.join(speaker_dir,'audio')
    os.mkdir(speaker_dir)
    os.mkdir(audio_dir)
    for path in train_grouped.get_group(speaker)['path']:
        utt_in = os.path.join(cv_dir,'clips',path+'.mp3')
        utt_out = os.path.join(audio_dir, path+'.wav')
        ffmpeg.input(utt_in).output(utt_out).run()
