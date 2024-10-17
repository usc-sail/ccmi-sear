# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

# the unified ast models for all pretraining/fine-tuning tasks.

import torch.nn as nn
import torch
import sys
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random
from .mlp_layers import MLPlayers
from .loss_fn import CLIPLoss1D
from .attention import DotProductAttention, CrossAttention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModelClip(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None):

        super(ASTModelClip, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=0)
            self.lsoftmax = nn.LogSoftmax(dim=0)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
            
            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

            # mlp head for fine-tuning
#            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
#                                          nn.Linear(self.original_embedding_dim, label_dim))

            #self.audio_transform = MLPlayers(units=[768, 512], dropout=0.1)
            #self.text_transform = MLPlayers(units=[512, 512], dropout=0.1)
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))

            self.audio_transform = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 512))
            self.text_transform = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
#            self.text_transform_attn = nn.ModuleList([
#                Block(
#                    dim=512, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
#                    drop=0, attn_drop=0, drop_path=0)
#                for i in range(2)])

            self.text_transform_attn = DotProductAttention(dim=512)
            self.text_transform_xattn = CrossAttention(dim=512)
#            self.tt_norm = nn.LayerNorm(512, eps=1e-6)
            self.loss_fn = CLIPLoss1D()
#            self.loss_fn_t = CLIPLoss1D()

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # generate permutation for 16*16 patch
    def gen_permid_patch(self, sequence_len=512, perm_size=100, blocksize=3):
        perm_id = torch.arange(sequence_len)
        if perm_size == 'random':
            perm_fraction = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            perm_size = round(perm_fraction*sequence_len)
        else:
            perm_size = int(perm_size)
        # randomize clutering factor in [3,6)
#       cur_clus = randrange(cluster) + 3
#        num_blocks = perm_size // (blockwidth ** 2)
        #bw, bh = [], []
        if blocksize == 'global':
            transform_ids = torch.tensor(random.sample(range(sequence_len), perm_size))
            perm_id[transform_ids] = transform_ids[torch.randperm(len(transform_ids))]
            num_shuf = torch.ne(perm_id, torch.arange(sequence_len)).sum()
            return perm_id, perm_size, num_shuf

        num_shuf = 0
        while num_shuf < perm_size - 10:
            start_id = randrange(sequence_len)
            blockwidth = randrange(2, self.p_t_dim//4) if blocksize=='random' else int(blocksize)
            blockheight = randrange(2, self.p_f_dim) if blocksize=='random' else int(blocksize)
         #   bw.append(blockwidth); bh.append(blockheight)
            transform_ids = [start_id + self.p_t_dim * i + j for i in range(blockheight) for j in range(blockwidth)]
            transform_ids = torch.tensor([x for x in transform_ids if x>0 and x<sequence_len])
            perm_id[transform_ids] = transform_ids[torch.randperm(len(transform_ids))]
            num_shuf = torch.ne(perm_id, torch.arange(sequence_len)).sum()
        return perm_id, perm_size, num_shuf


    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    def pretrain_clip(self, x, text, valid_lens=None, task=None):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        transformed_audio = self.audio_transform(x)
#        transformed_text = text
        if text == None:
            transformed_text = None
        elif task == 'pretrain_clip_attn':
            transformed_text = self.text_transform_attn(text, valid_lens)
            transformed_text = transformed_text.max(axis=1).values
        elif task == 'pretrain_clip_xattn':
            transformed_text = self.text_transform_xattn(text, transformed_audio, valid_lens)
            transformed_text = transformed_text.max(axis=1).values
        else:    
            transformed_text = self.text_transform(text)

        del x
        torch.cuda.empty_cache()
        return transformed_audio, transformed_text
    
    
    def pretrain_clip_masked(self, x, text, valid_lens, mask_patch, task):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        
        if self.cls_token_num == 2:
            audio_emb_clip = (x[:, 0] + x[:, 1])/2
        else:
            audio_emb_clip = x[:, 0]
        audio_emb_clip = self.audio_transform(audio_emb_clip)
        if task == 'pretrain_clip_attn_masked':
            transformed_text = self.text_transform_attn(text, valid_lens)
            text_emb_clip = transformed_text.max(axis=1)
        else:
            text_emb_clip = self.text_transform(text)
        #text_emb_clip = text

        # prediction of the masked patch
        pred_cont = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        pred_mse = torch.empty((B, mask_patch, self.fshape*self.tshape), device=x.device).float()
        #target = torch.empty((B, mask_patch, self.fshape*self.tshape), device=x.device).float()

        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred_cont[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            pred_mse[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
        
        mse = torch.mean((pred_mse - encode_samples) ** 2)
        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # equation (1) of the ssast paper
            total = torch.mm(encode_samples[i], torch.transpose(pred_cont[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)
        loss = mse*10 + nce
        return acc, loss, audio_emb_clip, text_emb_clip

            

    def joint_masked(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 17*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # prediction of the masked patch
        pred_cont = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        pred_mse = torch.empty((B, mask_patch, self.fshape*self.tshape), device=x.device).float()
        #target = torch.empty((B, mask_patch, self.fshape*self.tshape), device=x.device).float()

        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred_cont[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            pred_mse[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
        
        mse = torch.mean((pred_mse - encode_samples) ** 2)
        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # equation (1) of the ssast paper
            total = torch.mm(encode_samples[i], torch.transpose(pred_cont[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)
        loss = mse*10 + nce
        return acc, loss

    def forward(self, task, x, text, mask_patch=400, valid_lens=None):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        # alternatively, use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_clip' or task == 'pretrain_clip_attn' or task == 'pretrain_clip_xattn':
            return self.pretrain_clip(x, text, valid_lens, task)
        elif task == 'pretrain_clip_masked' or task=='pretrain_clip_attn_masked':
            return self.pretrain_clip_masked(x, text, valid_lens=valid_lens, mask_patch=mask_patch, task=task)
        else:
            raise Exception('Task unrecognized.')

if __name__ == '__main__':
    # this is an example of how to use the SSAST model

    # pretraining stage
    # suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
    input_tdim = 1024
    # create a 16*16 patch based AST model for pretraining.
    # note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
    ast_mdl = ASTModel(
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=True)
    # # alternatively, create a frame based AST model
    # ast_mdl = ASTModel(
    #              fshape=128, tshape=2, fstride=128, tstride=2,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain=True)

    # do pretraining, see src/traintest_mask.py for our full pretraining code
    # input in shape [batch_size, input_tdim, input_fdim]
    test_input = torch.zeros([10, input_tdim, 128])
    # mask 100 patches for both discriminative and generative loss
    acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
    loss = nce_loss + 10 * mse_loss
    # do back propagate and update the model, etc

    # after pretraining, save the pretrained model.
    # the code is designed for Dataparallel model
    ast_mdl = torch.nn.DataParallel(ast_mdl)
    torch.save(ast_mdl.state_dict(), './test_mdl.pth')

    # fine-tuning stage
    # now you have a labeled dataset you want to finetune AST on
    # suppose the avg length is 100 frames (1s) and there are 35 classes
    # the fshape and tshape must be same in pretraining and finetuning
    # but fstride and tstride can be different in pretraining and finetuning
    # using smaller strides improves the performance but also increase the computational overhead
    # set pretrain_stage as False since now is in the finetuning stage
    # provide the path of the pretrained model you want to load
    input_tdim = 100  # fine-tuning data length can be different with pretraining data length
    ast_mdl = ASTModel(label_dim=35,
                 fshape=16, tshape=16, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')
    # # alternatively, use a frame based AST model
    # ast_mdl = ASTModel(label_dim=35,
    #              fshape=128, tshape=2, fstride=128, tstride=1,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')

    # do finetuning, see src/traintest.py for our finetuning code
    test_input = torch.zeros([10, input_tdim, 128])
    prediction = ast_mdl(test_input, task='ft_avgtok')
    # output should in shape [batch_size, label_dim]
    print(prediction.shape)
    # calculate the loss, do back propagate, etc

    # # (optional) do some probe test
    # test_input = torch.zeros([1, input_tdim, 128]).to(device)
    # acc, nce = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    # # you can visualize the mask
    # pred, masked = ast_mdl(test_input, task='visualize_mask', mask_patch=100)
    # plt.imshow(masked[0,0])
    # plt.show()
