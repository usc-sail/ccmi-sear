# ccmi-sear
Public repository for SEAR audio model

## Dependencies
torch - 1.8.1  
torchaudio - 0.8.1  
timm - 0.4.5  
transformers - 4.17.0

## Training 
Install dependencies 
```
conda create -n env sear 
conda activate sear
pip install requirements.txt
```
Edit training parameters in run.sh
For baseline shown in the paper, the following parameters are used:
```
fshape=tshape=1     ## Convolution kernel-size for audio spectrogram patch embeddings
fstride=tstride=10   ## Convolution stride for audio spectrogram patch embeddings
mixup=0.5           ## Probability with which two samples and their labels will be mixed
freqm=48            ## Maximum Frequency Masking Strip
timem=192           ## Maximum Time Masking Strip
n_class=120         ## Number of sound classes
```


## Downstream
Run scripts under downstream/{task}/. Currently supports ESC-50 and Kinetics Sounds.


## Cite our work
```
@inproceedings{hebbar2023sear,
  title={SEAR: Semantically-grounded Audio Representations},
  author={Hebbar, Rajat and Bose, Digbalay and Narayanan, Shrikanth},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={2785--2794},
  year={2023}
}
```
