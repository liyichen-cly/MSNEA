import h5py
import numpy as np
np.set_printoptions(threshold=np.inf)
class KGs:
    def __init__(self, data_folder, division, ordered=True):
        kg1_relation_triples_set, kg1_entities_set, kg1_relations_set = self.read_relation_triples(data_folder + 'rel_triples_1')
        kg2_relation_triples_set, kg2_entities_set, kg2_relations_set = self.read_relation_triples(data_folder + 'rel_triples_2')

        self.ent_num = len(kg1_entities_set | kg2_entities_set)
        self.rel_num = len(kg1_relations_set | kg2_relations_set)

        self.ent_ids1_dict, self.ent_ids2_dict = self.generate_mapping_id(kg1_relation_triples_set, kg1_entities_set,
                                                    kg2_relation_triples_set, kg2_entities_set, ordered=ordered)
        self.rel_ids1_dict, self.rel_ids2_dict = self.generate_mapping_id(kg1_relation_triples_set, kg1_relations_set,
                                                    kg2_relation_triples_set, kg2_relations_set, ordered=ordered)
        self.ent_ids_dict = {**self.ent_ids1_dict, **self.ent_ids2_dict}
        self.id_relation_triples1, self.rt_dict1, self.hr_dict1 = self.uris_relation_triple_2ids(kg1_relation_triples_set, self.ent_ids1_dict, self.rel_ids1_dict)
        self.id_relation_triples2, self.rt_dict2, self.hr_dict2 = self.uris_relation_triple_2ids(kg2_relation_triples_set, self.ent_ids2_dict, self.rel_ids2_dict)
        self.kg1_entities_list = list(self.ent_ids1_dict.values())
        self.kg2_entities_list = list(self.ent_ids2_dict.values())

        self.relation_triples = self.id_relation_triples1.extend(self.id_relation_triples2)

        self.id_ent_images_res_dict1 = self.read_image_resnet(data_folder + 'resnet50_1.h5', self.ent_ids1_dict)
        self.id_ent_images_res_dict2 = self.read_image_resnet(data_folder + 'resnet50_2.h5', self.ent_ids2_dict)
        self.images_list = self.emerge_image_embedding(self.id_ent_images_res_dict1, self.id_ent_images_res_dict2)

        self.attr_list, self.attr_id_dict1, self.attr_id_dict2, \
        kg1_attribute_triples_set, kg2_attribute_triples_set = self.generate_attr_id(data_folder + 'attr_triples_1', data_folder + 'attr_triples_2',
                                                                                        data_folder + 'attr_name_1.h5', data_folder + 'attr_name_2.h5')
        _, self.eid_aid_v1 = self.uris_attribute_triple_2ids(kg1_attribute_triples_set, self.ent_ids1_dict, self.attr_id_dict1)
        _, self.eid_aid_v2 = self.uris_attribute_triple_2ids(kg2_attribute_triples_set, self.ent_ids2_dict, self.attr_id_dict2)


        self.train_links = self.read_links(data_folder + division + 'train_links', self.ent_ids1_dict, self.ent_ids2_dict)
        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]

        sup_triples1_set, sup_triples2_set = self.generate_sup_relation_triples(self.train_links,
                                                                self.rt_dict1, self.hr_dict1,
                                                                self.rt_dict2, self.hr_dict2)
        self.relation_triples_list1, self.relation_triples_list2 = self.add_sup_relation_triples(sup_triples1_set, sup_triples2_set)
        self.relation_triples_set1 = set(self.relation_triples_list1)
        self.relation_triples_set2 = set(self.relation_triples_list2)

        self.eid_aid_v_add1, self.eid_aid_v_add2 = self.add_sup_attribute_triples(self.train_links, self.eid_aid_v1, self.eid_aid_v2)
        self.attr_max_num, self.eid_aid_list, self.eid_vid_list, self.eav_len_list, self.eid_mask_list = self.generate_attr_list(self.eid_aid_v_add1, self.eid_aid_v_add2)
        
        self.test_links = self.read_links(data_folder + division + 'test_links', self.ent_ids1_dict, self.ent_ids2_dict)
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]
        self.valid_links = self.read_links(data_folder + division + 'valid_links', self.ent_ids1_dict, self.ent_ids2_dict)
        self.valid_entities1 = [link[0] for link in self.valid_links]
        self.valid_entities2 = [link[1] for link in self.valid_links]



    def read_relation_triples(self, file_path):
        print("read relation triples:", file_path)
        triples = set()
        entities, relations = set(), set()
        file = open(file_path, 'r', encoding='utf8')
        for line in file.readlines():
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = params[0].strip()
            r = params[1].strip()
            t = params[2].strip()
            triples.add((h, r, t))
            entities.add(h)
            entities.add(t)
            relations.add(r)
        file.close()
        return triples, entities, relations

    def generate_attr_id(self, att_f1, att_f2, att_embed_1, att_embed_2):
        id_attr_dict1 = dict()
        attr_id_dict1 = dict()
        id_attr_dict2 = dict()
        attr_id_dict2 = dict()
        triples1 = set()
        triples2 = set()
        attr_embed = []
        cnt = 1
        file = open(att_f1, 'r', encoding='utf-8')
        for line in file.readlines():
            params = line.strip().split('\t')
            assert len(params) == 3
            e = params[0].strip()
            a = params[1].strip()
            v = params[2].strip()
            triples1.add((e, a, v))
            if params[1] not in attr_id_dict1.keys():
                id_attr_dict1[cnt] = params[1]
                attr_id_dict1[params[1]] = cnt
                cnt+=1
        file.close()
        assert len(id_attr_dict1) == len(attr_id_dict1)

        file = open(att_f2, 'r', encoding='utf-8')
        for line in file.readlines():
            params = line.strip().split('\t')
            assert len(params) == 3
            e = params[0].strip()
            a = params[1].strip()
            v = params[2].strip()
            triples2.add((e, a, v))
            if params[1] not in attr_id_dict2.keys():
                id_attr_dict2[cnt] = params[1]
                attr_id_dict2[params[1]] = cnt
                cnt+=1
        file.close()
        assert len(id_attr_dict2) == len(attr_id_dict2)

        attr_num = len(id_attr_dict1) + len(id_attr_dict2)

        f1 = h5py.File(att_embed_1, 'r')
        f2 = h5py.File(att_embed_2, 'r')

        attr_embed.append(np.zeros(768))
        for i in range(1, attr_num + 1):
            if i in id_attr_dict1.keys():
                emb = np.array(f1[id_attr_dict1[i]])
                attr_embed.append(emb)
            elif i in id_attr_dict2.keys():
                emb = np.array(f2[id_attr_dict2[i]])
                attr_embed.append(emb)
            else:
                print("error!")
                exit()
        return attr_embed, attr_id_dict1, attr_id_dict2, triples1, triples2

    def read_links(self, file_path, e_id_1, e_id_2):
        print("read links:", file_path)
        links = list()
        file = open(file_path, 'r', encoding='utf8')
        for line in file.readlines():
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            e1 = e_id_1[params[0].strip()]
            e2 = e_id_2[params[1].strip()]
            links.append([e1, e2])
        return links

    def generate_sup_relation_triples_one_link(self, e1, e2, rt_dict, hr_dict):
        new_triples = set()
        for r, t in rt_dict.get(e1, set()):
            new_triples.add((e2, r, t))
        for h, r in hr_dict.get(e1, set()):
            new_triples.add((h, r, e2))
        return new_triples

    def generate_sup_relation_triples(self, sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
        new_triples1, new_triples2 = set(), set()
        for ent1, ent2 in sup_links:
            new_triples1 |= (self.generate_sup_relation_triples_one_link(ent1, ent2, rt_dict1, hr_dict1))
            new_triples2 |= (self.generate_sup_relation_triples_one_link(ent2, ent1, rt_dict2, hr_dict2))
        print("supervised relation triples: {}, {}".format(len(new_triples1), len(new_triples2)))
        return new_triples1, new_triples2

    def add_sup_relation_triples(self, sup_triples1, sup_triples2):
        id_relation_triples1_set = set(self.id_relation_triples1)
        id_relation_triples1_set |= sup_triples1
        id_relation_triples2_set = set(self.id_relation_triples2)
        id_relation_triples2_set |= sup_triples2
        return list(id_relation_triples1_set), list(id_relation_triples2_set)

    def add_sup_attribute_triples(self, sup_links, e_av1, e_av2):
        add_attr_num1 = 0
        add_attr_num2 = 0
        for e1, e2 in sup_links:
            sup_e1 = e_av2.get(e2, set())
            sup_e2 = e_av1.get(e1, set())
            new_attr_set = sup_e1 | sup_e2
            e_av1[e1] = new_attr_set
            e_av2[e2] = new_attr_set
            add_attr_num1 += len(sup_e2)
            add_attr_num2 += len(sup_e1)
        print("sup attribute triples: {}, {}".format(add_attr_num1, add_attr_num2))
        return e_av1, e_av2

    def generate_attr_list(self, e_av1, e_av2):
        max_num = 0
        cnt_zero = 0
        entid_attr_list = []
        entid_value_list = []
        entid_av_len_list = []

        for eid in range(self.ent_num):
            if eid in e_av1.keys():
                max_num = max(max_num, len(e_av1[eid]))
            elif eid in e_av2.keys():
                max_num = max(max_num, len(e_av2[eid]))
            else:
                cnt_zero += 1
        print("attribute max num: {}".format(max_num))

        av_mask = np.ones((self.ent_num, max_num))
        for eid in range(self.ent_num):
            if eid in e_av1.keys():
                av_l = len(e_av1[eid])
                av = e_av1[eid]
            elif eid in e_av2.keys():
                av_l = len(e_av2[eid])
                av = e_av2[eid]
            else:
                av_l = 0
                av = []
            a = [ea for (ea, _) in av]
            v = [ev for (_, ev) in av]
            for i in range(max_num - av_l):
                a.append(0)
                v.append(0)
            av_mask[eid][av_l:] = 0
            entid_attr_list.append(a)
            entid_value_list.append(v)
            entid_av_len_list.append(av_l)
        return max_num, entid_attr_list, entid_value_list, entid_av_len_list, av_mask.tolist()

    def sort_elements(self, triples, elements_set):
        dic = dict()
        for s, p, o in triples:
            if s in elements_set:
                dic[s] = dic.get(s, 0) + 1
            if p in elements_set:
                dic[p] = dic.get(p, 0) + 1
            if o in elements_set:
                dic[o] = dic.get(o, 0) + 1

        sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
        ordered_elements = [x[0] for x in sorted_list]
        return ordered_elements, dic

    def generate_mapping_id(self, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
        ids1, ids2 = dict(), dict()
        if ordered:
            kg1_ordered_elements, _ = self.sort_elements(kg1_triples, kg1_elements)
            kg2_ordered_elements, _ = self.sort_elements(kg2_triples, kg2_elements)
            n1 = len(kg1_ordered_elements)
            n2 = len(kg2_ordered_elements)
            n = max(n1, n2)
            for i in range(n):
                if i < n1 and i < n2:
                    ids1[kg1_ordered_elements[i]] = i * 2
                    ids2[kg2_ordered_elements[i]] = i * 2 + 1
                elif i >= n1:
                    ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
                else:
                    ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
        else:
            index = 0
            for ele in kg1_elements:
                if ele not in ids1:
                    ids1[ele] = index
                    index += 1
            for ele in kg2_elements:
                if ele not in ids2:
                    ids2[ele] = index
                    index += 1
        assert len(ids1) == len(set(kg1_elements))
        assert len(ids2) == len(set(kg2_elements))
        return ids1, ids2

    def uris_relation_triple_2ids(self, uris, ent_ids, rel_ids):
        id_uris = list()
        rt_dict, hr_dict = dict(), dict()
        for u1, u2, u3 in uris:
            assert u1 in ent_ids
            h_id = ent_ids[u1]
            assert u2 in rel_ids
            r_id = rel_ids[u2]
            assert u3 in ent_ids
            t_id = ent_ids[u3]
            id_uris.append((h_id, r_id, t_id))

            rt_set = rt_dict.get(h_id, set())
            rt_set.add((r_id, t_id))
            rt_dict[h_id] = rt_set

            hr_set = hr_dict.get(t_id, set())
            hr_set.add((h_id, r_id))
            hr_dict[t_id] = hr_set

        assert len(id_uris) == len(set(uris))
        return id_uris, rt_dict, hr_dict

    def uris_attribute_triple_2ids(self, uris, ent_ids, att_ids):
        id_uris = list()
        e_av_dict = dict()
        for u1, u2, u3 in uris:
            assert u1 in ent_ids
            e_id = ent_ids[u1]
            assert u2 in att_ids
            a_id = att_ids[u2]
            v = u3.split('\"^^')[0].strip('\"')
            if 'e-' in v:
                pass
            elif '-' in v and v[0]!='-':
                v = v.split('-')[0]
            elif v[0]=='-' and v.count('-')>1:
                v = '-' + v.split('-')[1]
            if '#' in v:
                v = v.strip('#')
            id_uris.append((e_id, a_id, float(v)))

            av_set = e_av_dict.get(e_id, set())
            av_set.add((a_id, float(v)))
            e_av_dict[e_id] = av_set

        assert len(id_uris) == len(set(uris))
        return id_uris, e_av_dict

    def emerge_image_embedding(self, entid_image1, entid_image2):
        image_embed = []
        num = len(entid_image1)+len(entid_image2)
        for i in range(num):
            if i in entid_image1.keys():
                image_embed.append(entid_image1[i])
            else:
                image_embed.append(entid_image2[i])
        return image_embed

    def read_image_resnet(self, h5_file, ent_ids):
        print("read image resnet:", h5_file)
        entid_image = dict()
        f = h5py.File(h5_file, 'r')
        for (ent, id) in ent_ids.items():
            if ent in f.keys():
                entid_image[id] = np.array(f[ent])
            else:
                entid_image[id] = np.zeros(2048)
        f.close()
        return entid_image

