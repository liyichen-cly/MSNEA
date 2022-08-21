import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from evaluation import *

class MSNEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.ent_num = kgs.ent_num
        self.rel_num = kgs.rel_num
        self.kgs = kgs
        self.args = args
        self.img_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kgs.images_list))
        self.ent_embed = nn.Embedding(self.ent_num, self.args.dim)
        self.rel_embed = nn.Embedding(self.rel_num, self.args.dim)
        nn.init.xavier_normal_(self.ent_embed.weight.data)
        nn.init.xavier_normal_(self.rel_embed.weight.data)
        self.fc1 = nn.Linear(2048, self.args.dim)
        self.fc2 = nn.Linear(2048, self.args.dim)
        self.fc3 = nn.Linear(2048, self.args.dim)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

        self.attr_encoder = AttrEncoder(self.kgs, self.args)
        self.fc_a = nn.Linear(1, self.args.dim)
        self.fc_a1 = nn.Linear(self.args.dim, self.args.dim)
        nn.init.xavier_normal_(self.fc_a.weight.data)
        nn.init.xavier_normal_(self.fc_a1.weight.data)

    def forward(self, p_h, p_r, p_t, n_h, n_r, n_t, e1, e2, \
                    e1_a, e1_v, e1_mask, e1_l, e2_a, e2_v, e2_mask, e2_l):
        r_p_h = self.r_rep(p_h)
        i_p_h = self.i_w(p_h)
        r_p_r = F.normalize(self.rel_embed(p_r), 2, -1)
        r_p_t = self.r_rep(p_t)
        i_p_t = self.i_w(p_t)

        r_n_h = self.r_rep(n_h)
        i_n_h = self.i_w(n_h)
        r_n_r = F.normalize(self.rel_embed(n_r), 2, -1)
        r_n_t = self.r_rep(n_t)
        i_n_t = self.i_w(n_t)

        pos_dis = r_p_h + r_p_r - r_p_t
        neg_dis = r_n_h + r_n_r - r_n_t

        pos_score = torch.sum(torch.square(pos_dis), dim=1)
        neg_score = torch.sum(torch.square(neg_dis), dim=1)
        relation_loss = torch.sum(F.relu(self.args.margin + pos_score - neg_score))
        pos_dis1 = i_p_h + r_p_r - i_p_t
        neg_dis1 = i_n_h + r_n_r - i_n_t
        pos_score1 = torch.sum(torch.square(pos_dis1), dim=1)
        neg_score1 = torch.sum(torch.square(neg_dis1), dim=1)
        relation_loss += torch.sum(F.relu(self.args.margin + pos_score1 - neg_score1))

        e1_r_embed = self.r_rep(e1)
        e2_r_embed = self.r_rep(e2)
        e1_i_embed = self.i_rep(e1)
        e2_i_embed = self.i_rep(e2)
        e1_a_embed = self.attr_encoder(e1_a, e1_v, e1_mask, e1_l, e1_i_embed)
        e2_a_embed = self.attr_encoder(e2_a, e2_v, e2_mask, e2_l, e2_i_embed)



        e1_all = self.fusion(e1_r_embed, e1_i_embed, e1_a_embed)
        e2_all = self.fusion(e2_r_embed, e2_i_embed, e2_a_embed)

        r_score = torch.mm(e1_r_embed, e2_r_embed.t())
        a_score = torch.mm(e1_a_embed, e2_a_embed.t())
        i_score = torch.mm(e1_i_embed, e2_i_embed.t())
        score = torch.mm(e1_all, e2_all.t())

        return relation_loss, r_score, a_score, i_score, score

    def predict(self, e1, e2, e1_a, e1_v, e1_mask, e1_l, e2_a, e2_v, e2_mask, e2_l):
        e1_r_embed = self.r_rep(e1)
        e2_r_embed = self.r_rep(e2)
        e1_i_embed = self.i_rep(e1)
        e2_i_embed = self.i_rep(e2)

        e1_a_embed = self.attr_encoder(e1_a, e1_v, e1_mask, e1_l, e1_i_embed)
        e2_a_embed = self.attr_encoder(e2_a, e2_v, e2_mask, e2_l, e2_i_embed)

        e1_all = self.fusion(e1_r_embed, e1_i_embed, e1_a_embed)
        e2_all = self.fusion(e2_r_embed, e2_i_embed, e2_a_embed)
        return e1_all.cpu().numpy(), e2_all.cpu().numpy(), \
                e1_r_embed.cpu().numpy(), e2_r_embed.cpu().numpy(), \
                e1_i_embed.cpu().numpy(), e2_i_embed.cpu().numpy(), \
                e1_a_embed.cpu().numpy(), e2_a_embed.cpu().numpy()

    def r_rep(self, e):
        return F.normalize(self.ent_embed(e), 2, -1)

    def i_rep(self, e):
        return F.normalize(self.fc1(self.img_embed(e)), 2, -1)

    def i_w(self, e):
        return F.normalize(self.fc3(self.img_embed(e)), 2, -1)

    def fusion(self, a, b, c):
        all = F.normalize(torch.cat([a, b, c], dim=1), 2, -1)
        return all


class AttrEncoder(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.args = args
        self.attr_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kgs.attr_list))
        self.fc1 = nn.Linear(768, self.args.dim)
        self.fc2 = nn.Linear(2*self.args.dim, self.args.dim)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, e_a, e_v, mask, l, i):
        e_a = self.fc1(self.attr_embed(e_a))
        e_v = torch.sigmoid(e_v.unsqueeze(-1)).repeat(1,1,self.args.dim)
        e = self.fc2(torch.cat([e_a, e_v], dim=2))
        alpha = F.softmax(torch.sum(e * i.unsqueeze(1), dim=-1) * mask, dim=1)
        e = torch.sum(alpha.unsqueeze(2) * e, dim=1)
        return e


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dis, label, margin=2.0):
        loss_contrastive = torch.mean((1-label) * torch.pow(dis, 2) +
                                      (label) * torch.pow(torch.clamp(margin - dis, min=0.0), 2))
        return loss_contrastive