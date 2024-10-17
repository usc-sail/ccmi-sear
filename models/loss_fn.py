import numpy as np
import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPLoss1D(nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()
#        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        batch_size = image_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        loss = (
            self.loss_image(logits_per_image, ground_truth)
            + self.loss_text(logits_per_text, ground_truth)
        ) / 2
        
        acc = (
            torch.eq(torch.argmax(logits_per_image, dim=-1), ground_truth).sum() / batch_size
            + torch.eq(torch.argmax(logits_per_text, dim=-1), ground_truth).sum() / batch_size
        ) / 2

        return acc, loss




class MILNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MILNCELoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) *np.log(1/temperature))

    def forward(self, video_embd, text_embd):
        logit_scale = self.logit_scale.exp()

        x = logit_scale * torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.tensor(0.0), torch.mean(denominator - nominator)
        
class MILNCEMaxLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MILNCEMaxLoss, self).__init__()   
        self.logit_scale = nn.Parameter(torch.ones([]) *np.log(1/temperature))

    def forward(self, video_embd, text_embd):
        logit_scale = self.logit_scale.exp()
        x = logit_scale * torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.log(torch.max(torch.exp(nominator), dim=1).values)
#        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.tensor(0.0), torch.mean(denominator - nominator)
        
