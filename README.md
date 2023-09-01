# Unsupervised Hashing with Similarity Distribution Calibration

[ArXiv](https://arxiv.org/abs/2302.07669) | <a href="https://github.com/kamwoh/sdc/blob/master/docs/suppmat.pdf">
Supplementary Material</a>

### Official pytorch implementation of the paper: "Unsupervised Hashing with Similarity Distribution Calibration"

#### BMVC 2023

# Description

Unsupervised hashing methods typically aim to preserve the similarity between data points in a feature space by mapping
them to binary hash codes. However, these methods often overlook the fact that the similarity between data points in the
continuous feature space may not be preserved in the discrete hash code space, due to the limited similarity range of
hash codes.
The similarity range is bounded by the code length and can lead to a problem known as _similarity collapse_. That is,
the positive and negative pairs of data points become less distinguishable from each other in the hash space.
To alleviate this problem, in this paper a novel Simialrity Distribution Calibration (SDC) method is introduced.
SDC aligns the hash code similarity distribution towards a calibration distribution (e.g., beta distribution) with
sufficient spread across the entire similarity range, thus alleviating the similarity collapse problem.
Extensive experiments show that our SDC outperforms significantly the state-of-the-art alternatives on coarse
category-level and instance-level image retrieval.

# How to run

### Training

```bash
python main_v2.py model=sdc model.nbit=64 dataset=cifar10 
```

The above command runs cifar10 with 64-bits settings. We use [hydra](https://hydra.cc/docs/intro/) as config system.

### Testing

```bash
python val.py model=sdc logdir=/path/to/logdir
```

The above command runs validation for the trained model, please see `val.py` to have different validation parameters.

# Dataset

Please refer to https://fast-image-retrieval.readthedocs.io/en/latest/dataset.html for dataset downloading.

# Feedback

Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by
sending an email to `kamwoh.ng at surrey.ac.uk`.

# Related Works

1. Deep Polarized Network (DPN) - (https://github.com/kamwoh/DPN)
2. One Loss for All (OrthoHash) - (https://github.com/kamwoh/orthohash)
3. Fast Image Retrieval (FIRe) - (https://github.com/CISiPLab/cisip-FIRe)

# Implementations

We also implemented (some copied from original authors):

1. Unsupervised GreedyHash
2. BiHalf
3. Weighted Contrastive Hashing
4. Naturally Sort Hash
5. CIBHash

See `models/arch`, `models/trainers` and `models/loss`.

# Citation

If you find this work useful for your research, please cite

```bibtex
@inproceedings{ng2023sdc,
  author = {Ng, Kam Woh and Zhu, Xiatian and Hoe, Jiun Tian and Chan, Chee Seng and Zhang, Tianyu and Song, Yi-Zhe and Xiang, Tao},
  booktitle = {British Machine Vision Conference}, 
  title = {Unsupervised Hashing with Similarity Distribution Calibration}, 
  year = {2023}
}
```
