import os
import time
import json
import multiprocessing as mp
import torch
import math
import logging
import matplotlib.pyplot as plt
import random
from utils import *
from evaluation import *
from kgs import *
from model import MSNEA, ContrastiveLoss
plt.switch_backend('agg')

def load_args(file):
    with open(file, 'r') as f:
        args_dict = json.load(f)
    for (k, v) in args_dict.items():
        logger.info(str(k)+' : '+str(v))
    args = ARGs(args_dict)
    return args

class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

def set_logger():
    filename = './logs/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log'
    logger = logging.getLogger(filename)
    format_str = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = logging.FileHandler(filename, mode='a', encoding='utf-8', delay=False)
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger
logger = set_logger()


def test(model, kgs, args, save=True):
    model.eval()
    with torch.no_grad():
        e1 = torch.LongTensor(kgs.test_entities1).cuda()
        e2 = torch.LongTensor(kgs.test_entities2).cuda()
        e1_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.test_entities1]).cuda()
        e1_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.test_entities1]).cuda()
        e2_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.test_entities2]).cuda()
        e2_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.test_entities2]).cuda()
        mask1 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.test_entities1]).cuda()
        mask2 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.test_entities2]).cuda()
        l1 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.test_entities1]).cuda()
        l2 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.test_entities2]).cuda()
        embeds1, embeds2, e_r1, e_r2, e_i1, e_i2, e_a1, e_a2 = model.predict(e1, e2, e1_attr, e1_val, mask1, l1, e2_attr, e2_val, mask2, l2)
        rest, hits1, mr, all_mrr = greedy_alignment(logger, embeds1, embeds2, args.top_k, args.test_threads_num,
                                metric=args.eval_metric, normalize=args.eval_norm, csls_k=0, accurate=True)
    return all_mrr

def valid(model, kgs, args):
    model.eval()
    with torch.no_grad():
        e1 = torch.LongTensor(kgs.valid_entities1).cuda()
        e2 = torch.LongTensor(kgs.valid_entities2).cuda()
        e1_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.valid_entities1]).cuda()
        e1_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.valid_entities1]).cuda()
        e2_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.valid_entities2]).cuda()
        e2_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.valid_entities2]).cuda()
        mask1 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.valid_entities1]).cuda()
        mask2 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.valid_entities2]).cuda()
        l1 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.valid_entities1]).cuda()
        l2 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.valid_entities2]).cuda()
        embeds1, embeds2, e_r1, e_r2, e_i1, e_i2, e_a1, e_a2  = model.predict(e1, e2, e1_attr, e1_val, mask1, l1, e2_attr, e2_val, mask2, l2)
        _, hits1, mr, mrr = greedy_alignment(logger, embeds1, embeds2, args.top_k, args.test_threads_num,
                                                        args.eval_metric, args.eval_norm, csls_k=0, accurate=False)
    return hits1 if args.stop_metric == 'hits1' else mrr

def train(model, kgs, args, out_folder):
    t = time.time()
    relation_triples_num = len(kgs.relation_triples_list1) + len(kgs.relation_triples_list2)
    relation_triple_steps = int(math.ceil(relation_triples_num / args.batch_size))
    relation_step_tasks = task_divide(list(range(relation_triple_steps)), args.batch_threads_num)
    flag1, flag2 = -1, -1
    manager = mp.Manager()
    relation_batch_queue = manager.Queue()
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    align_criterion = ContrastiveLoss()

    train_e1 = torch.LongTensor(kgs.train_entities1).cuda()
    train_e2 = torch.LongTensor(kgs.train_entities2).cuda()
    e1_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.train_entities1]).cuda()
    e1_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.train_entities1]).cuda()
    e2_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.train_entities2]).cuda()
    e2_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.train_entities2]).cuda()
    mask1 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.train_entities1]).cuda()
    mask2 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.train_entities2]).cuda()
    l1 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.train_entities1]).cuda()
    l2 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.train_entities2]).cuda()
    label_ = torch.eye(len(kgs.train_entities1)).cuda()
    max_mrr = 0
    for i in range(1, args.max_epoch + 1):
        start = time.time()
        epoch_loss = 0
        epoch_rloss = 0
        epoch_closs = 0
        trained_samples_num = 0
        model.train()
        for steps_task in relation_step_tasks:
            mp.Process(target=generate_relation_triple_batch_queue,
                        args=(kgs.relation_triples_list1, kgs.relation_triples_list2,
                                kgs.relation_triples_set1, kgs.relation_triples_set2,
                                kgs.kg1_entities_list, kgs.kg2_entities_list,
                                args.batch_size, steps_task,
                                relation_batch_queue, args.neg_triple_num)).start()
        for _ in range(relation_triple_steps):
            optimizer.zero_grad()
            batch_pos, batch_neg = relation_batch_queue.get()
            rel_p_h = torch.LongTensor([x[0] for x in batch_pos]).cuda()
            rel_p_r = torch.LongTensor([x[1] for x in batch_pos]).cuda()
            rel_p_t = torch.LongTensor([x[2] for x in batch_pos]).cuda()
            rel_n_h = torch.LongTensor([x[0] for x in batch_neg]).cuda()
            rel_n_r = torch.LongTensor([x[1] for x in batch_neg]).cuda()
            rel_n_t = torch.LongTensor([x[2] for x in batch_neg]).cuda()

            r_loss, rs, ats, ims, score = model(rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t, \
                                    train_e1, train_e2, e1_attr, e1_val, mask1, l1, e2_attr, e2_val, mask2, l2)
            align_loss = align_criterion(score, label_) + align_criterion(rs, label_) + align_criterion(ats, label_) + align_criterion(ims, label_)
            loss = r_loss + align_loss
            loss.backward()
            optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += loss.item()
            epoch_rloss += r_loss.item()
            epoch_closs += align_loss.item()

        epoch_loss /= trained_samples_num
        epoch_rloss /= trained_samples_num
        epoch_closs /= len(kgs.train_entities1)
        random.shuffle(kgs.relation_triples_list1)
        random.shuffle(kgs.relation_triples_list2)
        end = time.time()
        logger.info('[epoch {}] loss: {:.6f}, relation loss: {:.6f}, align loss:{:.6f}, time: {:.4f}s'.format(i, epoch_loss, epoch_rloss, epoch_closs, end - start))
        loss_list.append(epoch_loss)

        if i >= args.start_valid and i % args.eval_freq == 0:
            flag = valid(model, kgs, args)
            if flag > max_mrr:
                torch.save(model.state_dict(), out_folder + 'model_best.pkl')
                max_mrr = flag
            flag1, flag2, stop = early_stop(flag1, flag2, flag)
            if args.early_stop and (stop or i == args.max_epoch):
                print("\n == should early stop == \n")
                break

    plt.plot(range(len(loss_list)), loss_list)
    plt.savefig("train_loss.png")
    logger.info("Training ends. Total time = {:.3f} s.".format(time.time() - t))

def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    print("results output folder:", folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

if __name__ == '__main__':
    t = time.time()
    args = load_args('config.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    out_folder = generate_out_folder(args.output, args.training_data, args.dataset_division, 'MSNEA')
    kgs = KGs(args.training_data, args.dataset_division, ordered=True)
    model = MSNEA(kgs, args)
    model.cuda()
    train(model, kgs, args, out_folder)
    model.load_state_dict(torch.load(out_folder + 'model_best.pkl'))
    mrr = test(model, kgs, args)
    logger.info("Total run time = {:.3f} s.".format(time.time() - t))

