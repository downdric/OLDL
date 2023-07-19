# Introduction

	This is the implementation of ICCV 2023 paper "Ordinal Label Distribution Learning".

# Abstract

Label distribution learning (LDL) is a recent hot topic, in which ambiguity is modeled via the description degree of each label. However, in common LDL tasks, e.g., age estimation, labels are in an intrinsic order. The conventional LDL paradigm adopts a per-label manner for optimization, neglecting the internal sequential patterns of labels. Therefore, we propose a new paradigm, termed ordinal label distribution learning (OLDL).
We model the sequential patterns of labels from aspects of spatial, semantic, and temporal order relationships. The spatial order depicts the relative position between arbitrary labels. We build cross-label transformation between distributions, which is determined by the spatial margin in labels. Labels naturally yield different semantics, so the semantic order is represented by constructing semantic correlations between arbitrary labels. The temporal order describes that the presence of labels is determined by their order, i.e. five after four. The value of a particular label contains information about previous labels, and we adopt cumulative distribution to construct this relationship. Based on these characteristics of ordinal labels, we propose the learning objectives and evaluation metrics for OLDL, namely CAD, QFD, and CJS. Comprehensive experiments conducted on four tasks demonstrate the superiority of OLDL against other existing LDL methods in both traditional and newly proposed metrics.


# Implementation

```
- install pytorch == 1.13.0
- download data and modify the "data_path" in data_set.py
- use dataset_name.py to train the network
- we use the evaluation tools provided by [Geng Xin]("http://palm.seu.edu.cn/xgeng/LDL/resource/LDLPackage_v1.2.zip")
```



