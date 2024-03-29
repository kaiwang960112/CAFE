## CAFE: Learning to Condense Dataset by Aligning Features [link](https://arxiv.org/pdf/2203.01531.pdf)

### This is a method of dataset condensation, and it has been accepted by CVPR-2022.



                              Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Shuo Yang, 
			      Shuo Wang, Guan Huang, Hakan Bilen, Xinchao Wang, and Yang You
                  National University of Singapore, The University of Edinburgh, PhiGent Robotics, 
	            University of Technology Sydney, Institute of Automation, Chinese Academy of Sciences
                                             kai.wang@comp.nus.edu.sg
			        Kai Wang and Bo Zhao are equally-contributted authors

## Getting Started
Run the following command to perform the distillation
```
python distill.py  --dataset CIFAR10  --model ConvNet  --ipc 10
```

### Abstract

Dataset condensation aims at reducing the network training effort through condensing a cumbersome training set into a compact synthetic one. State-of-the-art approaches largely rely on learning the synthetic data by matching the gradients between the realand synthetic data batches. Despite the intuitive motivation and promising results, such gradient-based methods, by nature, easily over-fit to a biased set of samples that produce dominant gradients, and thus lack a global supervision of data distribution. In this paper, we propose a novel scheme  to Condense dataset by Aligning FEatures (CAFE), which explicitly attempts to preserve 
the real-feature distribution as well as the discriminant power of the resulting  synthetic set, lending itself to strong generalization capability to various architectures. At the heart of our approach is an effective strategy to align features from the real and synthetic data across various scales, while accounting 
for the classification of real samples. Our scheme is further backed up by a novel dynamic bi-level optimization, which adaptively adjusts parameter updates to prevent over-/under-fitting. We validate the proposed CAFE across various datasets, and demonstrate that it generally outperforms the state of the art: on the SVHN dataset, for example, the performance gain is up to 11\%. Extensive experiments and analysis verify the effectiveness and necessity of proposed designs. Our code will be made publicly available. 
	
### Motivation
![image](https://github.com/kaiwang960112/CAFE/blob/main/figs/motivation.png)


### Pipeline
![image](https://github.com/kaiwang960112/CAFE/blob/main/figs/pipeline.png)

### Visualization
![image](https://github.com/kaiwang960112/CAFE/blob/main/figs/visualization.png)




