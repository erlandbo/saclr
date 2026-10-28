import torch
from torch import nn
import torchvision
from torch.nn import functional as F
import time



class SACLR1Cosine(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLR1Cosine, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.single_s = single_s
        self.temp = temp

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a 
        E_attr_b = q_attr_b 

        E_rep_a = q_rep_a 
        E_rep_b = q_rep_b 

        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B 

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a  
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats_a, feats_b, feats_idx):
        B = feats_a.shape[0]
        feats_a = F.normalize(feats_a, p=2, dim=1) 
        feats_b = F.normalize(feats_b, p=2, dim=1) 

        q_attr_a = torch.exp( torch.sum(feats_a * feats_b, dim=1) / self.temp )  
        q_attr_b = torch.exp( torch.sum(feats_b * feats_a, dim=1) / self.temp )  

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)
        
        neg_idxs = torch.roll(torch.arange(B), shifts=-1, dims=0)
        q_rep_a = torch.exp( torch.sum(feats_a * feats_b[neg_idxs], dim=1) / self.temp)  
        q_rep_b = torch.exp( torch.sum(feats_b * feats_a[neg_idxs], dim=1) / self.temp)  

        if self.single_s:
            feats_idx = 0

        Z_hat = self.s_inv[feats_idx] / self.N.pow(2)

        repulsive_forces_a = q_rep_a / Z_hat
        repulsive_forces_b = q_rep_b / Z_hat

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss



class NTXent(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp

    def forward(self, hidden1, hidden2, idx):
        
        LARGE_NUM = 1e9
        batch_size = hidden1.shape[0]

        hidden1 = F.normalize(hidden1, dim=1, p=2)
        hidden2 = F.normalize(hidden2, dim=1, p=2)
        
        masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()

        logits_aa = torch.matmul(hidden1, hidden1.T) / self.temp
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2.T) / self.temp
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2.T) / self.temp
        logits_ba = torch.matmul(hidden2, hidden1.T) / self.temp


        num_a = torch.exp(torch.diag(logits_ab) )
        num_b = torch.exp(torch.diag(logits_ba) )
        denom_a = torch.exp(torch.cat([logits_ab, logits_aa], dim=1) )
        denom_b = torch.exp(torch.cat([logits_ba, logits_bb], dim=1) )


        loss_a = -torch.log(num_a / torch.sum(denom_a, dim=1)).mean()
        loss_b = -torch.log(num_b / torch.sum(denom_b, dim=1)).mean()
        loss = (loss_a + loss_b) / 2.0
        
        return loss


B = 256


#criterion = NTXent(temp=0.5).cuda()
criterion = SACLR1Cosine(N=50_000, rho=0.999, alpha=0.125, s_init=50_000**(-2.0) * 10**0.0, single_s=True, temp=0.5).cuda()

print(criterion, "Batch={}".format(B))

torch.cuda.empty_cache()  # Empty cache, just in case PyTorch didn't do it (which is usually the case)

out1 = torch.randn((B, 1000), requires_grad=False).cuda()
out2 = torch.randn((B, 1000), requires_grad=False).cuda()

mem_activations = torch.cuda.memory_allocated()

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()  # Empty cache, just in case PyTorch didn't do it (which is usually the case)

with torch.no_grad():
    loss = criterion(out1, out2, 0)

mem_loss = torch.cuda.max_memory_allocated()



print("max_memory_loss", mem_loss )
print("memory-activ", mem_activations)
print("max-mem subtract activations", mem_loss - mem_activations)


