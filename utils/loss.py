import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(F.log_softmax(input, 1) *
                           (one_hot.detach())) / input.size(0)
        return loss


class ClipInfoCELoss(_Loss):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()
        
    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss, labels


def D(p, z):
    z = z.detach() # stop gradient
    p = p / p.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    return (p * z).sum(dim=1).mean() 

def D_minimize(p, z):  
    z = z.detach()
    p = p / p.norm(dim=-1, keepdim=True)
    z = (z / z.norm(dim=-1, keepdim=True)).permute(0, 2, 1)
    sim = torch.bmm(p, z)
    return sim.max(dim=-1)[0].mean(dim=-1).mean()


class SimsiamLoss(nn.Module):
    def __init__(self, symmetry=True):
        super(SimsiamLoss, self).__init__()
        self.symmetry = symmetry

    def forward(self, p1, z1, p2, z2, minimize_loss=False,):
        if self.symmetry:
            if minimize_loss:
                D1 = D_minimize(p1, z2)
                D2 = D_minimize(p2, z1)
                return -0.5 * (D1.mean() + D2.mean())
            else:
                D1 = D(p1, z2)
                D2 = D(p2, z1)
                return -0.5 * (D(p1, z2)  + D(p2, z1) )
            
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Tensor, shape [batch_size, embedding_dim]
            z_j: Tensor, shape [batch_size, embedding_dim]
        
        Returns:
            contrastive loss
        """
        batch_size = z_i.shape[0]

        representations = torch.cat([z_i, z_j], dim=0)  
        similarity_matrix = torch.matmul(representations, representations.T)  

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))

        positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        positive_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool).to(z_i.device)
        positive_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool).to(z_i.device)
        
        positive_samples_upper = torch.diagonal(similarity_matrix, offset=batch_size)  
        positive_samples_lower = torch.diagonal(similarity_matrix, offset=-batch_size) 

        positive_samples = torch.cat([positive_samples_upper, positive_samples_lower], dim=0)  

        negative_samples = similarity_matrix[~torch.logical_or(mask, positive_mask)].view(2 * batch_size, -1) 

        logits = torch.cat([positive_samples.unsqueeze(1), negative_samples], dim=1)  

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        loss = F.cross_entropy(logits / self.temperature, labels)

        return loss
