import torch
from torch.nn import functional as F
import torch.nn as nn



class STM_head(nn.Module):

    def __init__(self, ):
        super(STM_head, self).__init__()
        #self.conv1 = nn.Conv2d(84, 3, kernel_size=3, padding=1)

    def forward(self, q_key, q_val, m_key, m_val):
        N_q, C_q, H_q, W_q = q_key.shape
        N_m, C_m, T_m, H_m, W_m = m_key.shape

        q_key = q_key.view(N_q, C_q, -1) 
        m_key = m_key.view(N_m, C_m, -1)
        key_map = torch.einsum("ncq,ncm->nqm", (q_key, m_key))
        #key_map = torch.einsum("ncm,ncq->nmq", (m_key, q_key))

        #q_key = q_key.view(N_q,H_q*W_q, -1) 
        #m_key = m_key.view(N_m, -1, T_m*H_m*W_m)
        #key_map = torch.bmm(q_key, m_key) 

        key_map = key_map * ( C_q ** -0.5)
        #key_map = F.softmax(key_map, dim=2)
        key_map = F.softmax(key_map, dim=1)
        #key_map = F.softmax(torch.bmm(q_key,m_key), dim=1)
        m_val = m_val.view(N_m, -1, T_m*H_m*W_m)
        #m_val = m_val.view(N_m, T_m*H_m*W_m, -1)
        #mapped_feature = torch.bmm(key_map, m_val).view(N_m, -1, H_m, W_m)
        
        # N, C, HW,
        mapped_feature = torch.einsum("nqm,ncm->ncq", (key_map, m_val)).view(N_m, -1, H_m, W_m)
        #mapped_feature = torch.bmm(key_map, m_val).view(N_m, -1, H_m, W_m)

        out = torch.cat([q_val, mapped_feature], dim=1)

        return out


if __name__ == '__main__':
    q_key = torch.FloatTensor(2, 2, 2, 5)
    q_val = torch.FloatTensor(2, 8, 2, 5)
    #                    N, T, C, H, W
    m_key = torch.FloatTensor(2,3, 2 ,2, 5)
    m_val = torch.FloatTensor(2,3, 8 ,2, 5)
    stm = STM()
    o = stm(q_key, q_val, m_key, m_val)
    print(o.shape)

