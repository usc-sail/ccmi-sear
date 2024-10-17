import sys
import os, pickle, torch, numpy as np, pandas as pd, json
from torch.nn import functional as F
from transformers import CLIPTokenizer, CLIPModel
sys.path.append('/scratch1/rajatheb/ssl_icassp/')
from models import ASTModelClip
import pandas as pd
from dataloader_zs import AudioDataset, make_index_dict
os.environ['CUDA_VISIBLE_DEVICES']='0'

mean = -5.444231 
std = 3.2999249
target_length = 1024
json_file = './data/test.json'
csv_file = './data/kinetics_sounds_classes.csv'
label_prompt = ' '

pretrain_model = '/scratch1/rajatheb/ssl_icassp/pretrained_models/combined/ast-comb-clip-b180-lr1e-4-w12k.pth'
pretrain_name = pretrain_model.split('/')[-1].split('.pth')[0]
exp_dir = os.path.join('exp_zeroshot', pretrain_name)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#embedding_dir = '/data/rajatheb/ssl_movies/downstream/lvu-scene/data_clips/ast-base-clip-b90-w12k-decay0.9_cls/'
#exp_dir = './exp_ft_cls/ast-comb-clip-b180-lr1e-4-w12k_hlr3/'

audio_conf = {'mean': mean, 'std': std, 'target_length': target_length,
        'num_mel_bins':128, 'freqm':0, 'timem':0, 'mixup':0, 'noise': False, 'dataset': 'zs', 'mode':'evaluation'}
index_dict = make_index_dict(csv_file)


data_loader = torch.utils.data.DataLoader(
    AudioDataset(json_file, label_csv=csv_file, audio_conf=audio_conf),
                            batch_size=32, shuffle=False, num_workers=8, pin_memory=False)


audio_model = ASTModelClip(label_dim=len(index_dict), fshape=16, tshape=16, fstride=16, tstride=16, input_fdim=128, input_tdim=target_length, pretrain_stage=True, model_size='base')
sd = torch.load(pretrain_model)
if 'attn' not in pretrain_model:
    sd['module.text_transform_xattn.kv.weight'] = torch.rand(audio_model.text_transform_xattn.kv.weight.shape)
    sd['module.text_transform_xattn.q.weight'] = torch.rand(audio_model.text_transform_xattn.q.weight.shape)
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

audio_model.load_state_dict(sd, strict=False)
audio_model = audio_model.to(device)
audio_model.eval()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

index_dict_inv = {int(v):k for k,v in index_dict.items()}
index_dict_prompt = {k: label_prompt + v for k,v in index_dict_inv.items()}
inputs = tokenizer(list(index_dict_prompt.values()), padding=True, return_tensors='pt')
clip_feats = model.get_text_features(**inputs)
if 'attn' in pretrain_model:
    text_embs = audio_model.module.text_transform_attn(clip_feats.to('cuda').unsqueeze(0), valid_lens=None).squeeze(0)
else:
    text_embs = audio_model.module.text_transform(clip_feats.to('cuda'))

print("text embs shape", text_embs.shape)
text_embs = dict(zip(index_dict_prompt.keys(), text_embs))


audio_embs = {}
audio_labels = {}

with torch.no_grad():
    for i, (fileid, audio_input, labels) in enumerate(data_loader):
        audio_input = audio_input.to(device)
        audio_output, _ = audio_model('pretrain_clip', audio_input, text=None, valid_lens=None)
        audio_embs.update(dict(zip(fileid, audio_output)))        
        audio_labels.update(dict(zip(fileid, labels.argmax(1))))

file_ids = list(audio_embs.keys())
audio_embs = torch.stack([audio_embs[k] for k in file_ids])
audio_embs = F.normalize(audio_embs)

text_classes = list(text_embs.keys())
text_embs = torch.stack([text_embs[k] for k in text_classes])
text_embs = F.normalize(text_embs)

sim_mat = audio_embs @ text_embs.t()
preds = sim_mat.max(1).indices

labels = [audio_labels[k] for k in file_ids]
accuracy = sum([labels[i] == preds[i] for i in range(len(preds))])/len(preds)
print("top-1 accuracy, ", accuracy)

print(text_classes, index_dict)
for clidx in range(len(text_classes)):
    class_preds = [preds[i]==clidx for i in range(len(preds)) if labels[i] == clidx]
    print(f"{index_dict_inv[clidx]}: {sum(class_preds)/len(class_preds)}")

## Top-5 accuracy
preds_top5 = sim_mat.argsort(1)[:,-5:]
acc_top5 = sum([labels[i] in preds_top5[i] for  i in range(len(preds_top5))])/len(preds_top5)
print("top-5 accuracy, ", acc_top5)
for clidx in range(len(text_classes)):
    class_preds = [clidx in preds_top5[i] for i in range(len(preds_top5)) if labels[i] == clidx]
    print(f"{index_dict_inv[clidx]}: {sum(class_preds)/len(class_preds)}")



