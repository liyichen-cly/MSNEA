import random
import time
import numpy as np

def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    print("results output folder:", folder)
    return folder

def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks

def generate_relation_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2,
                                         entity_list1, entity_list2, batch_size,
                                         steps, out_queue, neg_triples_num):
    for step in steps:
        pos_batch, neg_batch = generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                              entity_list1, entity_list2, batch_size,
                                                              step, neg_triples_num)
        out_queue.put((pos_batch, neg_batch))
    exit(0)

def generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                   entity_list1, entity_list2, batch_size, step, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step)

    neg_batch1 = generate_neg_triples_fast(pos_batch1, triple_set1, entity_list1, neg_triples_num)
    neg_batch2 = generate_neg_triples_fast(pos_batch2, triple_set2, entity_list2, neg_triples_num)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2

def generate_pos_triples(triples, batch_size, step, is_fixed_size=False):
    start = step * batch_size
    end = start + batch_size
    if end > len(triples):
        end = len(triples)
    pos_batch = triples[start: end]
    if is_fixed_size and len(pos_batch) < batch_size:
        pos_batch += triples[:batch_size-len(pos_batch)]
    return pos_batch

def generate_neg_triples_fast(pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, relation, tail in pos_batch:
        neg_triples = list()
        nums_to_sample = neg_triples_num
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                i_neg_triples = list(i_neg_triples - all_triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == neg_triples_num:
                break
            else:
                nums_to_sample = neg_triples_num - len(neg_triples)
        assert len(neg_triples) == neg_triples_num
        neg_batch.extend(neg_triples)
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch