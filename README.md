# Multi-modal Siamese Network for Entity Alignment
Codes for paper "Multi-modal Siamese Network for Entity Alignment" published in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'2022).

## Dataset
Three public multi-modal knowledge graphs with **relational**, **attribute** and **visual** knowledge from paper "[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485)", i.e., FB15K, DB15K and YG15K.
There are **sameAs** links between FB15K and DB15K as well as between FB15K and YG15K, which could be regarded as **alignment** relations. 
[Please click here to download the datasets.](https://github.com/nle-ml/mmkb)

## Code
Our codes were modified based on the public benchmark [OpenEA](https://github.com/nju-websoft/OpenEA). We appreciate the authors for making OpenEA open-sourced.

## Citation
If you use this model or code, please kindly cite it as follows:
```
@inproceedings{chen2022multi,
  title={Multi-modal Siamese Network for Entity Alignment},
  author={Chen, Liyi and Li, Zhi and Xu, Tong and Wu, Han and Wang, Zhefeng and Yuan, Nicholas Jing and Chen, Enhong},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={118--126},
  year={2022}
}
```