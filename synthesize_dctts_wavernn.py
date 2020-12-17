import torch
from models.fatchord_version import WaveRNN
import argparse
import numpy as np
import librosa
import os
import sys
from scipy import signal
import zipfile

# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#

bits = 9                            # bit depth of signal
sample_rate = 22050
num_mels = 80
hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below

# Model Hparams
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches

# ------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__" :

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('-i', type=str, dest='inputdir', required=True, help='[string/path] Directory containing mag files')
    parser.add_argument('-o', type=str, dest='outputdir', required=True, help='[string/path] Output directory to store wavefiles')
    args   = parser.parse_args()
    inputdir = args.inputdir
    outputdir = args.outputdir

    os.makedirs('quick_start/voc_weights/', exist_ok=True)
    os.makedirs(outputdir, exist_ok=True)

    zip_ref = zipfile.ZipFile('pretrained/ljspeech.wavernn.mol.800k.zip', 'r')
    zip_ref.extractall('quick_start/voc_weights/')
    zip_ref.close()

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=voc_rnn_dims,
                        fc_dims=voc_fc_dims,
                        bits=bits,
                        pad=voc_pad,
                        upsample_factors=voc_upsample_factors,
                        feat_dims=num_mels,
                        compute_dims=voc_compute_dims,
                        res_out_dims=voc_res_out_dims,
                        res_blocks=voc_res_blocks,
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        mode=voc_mode)

    voc_model.load('quick_start/voc_weights/latest_weights.pyt')

    voc_k = voc_model.get_step() // 1000

    preemphasis = .97
    wfreq, hfilter = signal.freqz(b=[1],a=[1, -preemphasis],worN=1025,include_nyquist=True)
    hfilter = np.diag( abs(hfilter) )

    for filename in os.listdir(inputdir):

        if filename.endswith(".npy"):

            print(filename)

            mag = np.load( inputdir + "/%s"%filename)
            mag = mag.T

            mag = (np.clip(mag, 0, 1) * 100) - 100 + 20
            mag = np.power(10.0, mag * 0.05)
            mag = np.dot( hfilter , mag)
            mel = librosa.feature.melspectrogram(S=mag, sr=22050, n_fft=2048, n_mels=80, fmin=40)
            mel = 20* np.log10(np.maximum(1e-5, mel))
            mel = np.clip((mel  + 100) / 100,0,1)

            basename = filename[:-4]+'.wav'

            save_path = outputdir + '/%s'%basename

            m = torch.tensor(mel).unsqueeze(0)

            voc_model.generate(m, save_path, voc_gen_batched, voc_target, voc_overlap, mu_law)

            print("\n")


