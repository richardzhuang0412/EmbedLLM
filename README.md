# README

(Repo Under Construction!)

This repo contains the code for the experiments in our [paper](https://arxiv.org/abs/2410.02223): 

**EmbedLLM: Learning Compact Representations of Large Language Models**

By [Richard Zhuang](https://richardzhuang0412.github.io/), 
[Tianhao Wu](https://thwu1.github.io/tianhaowu/),
[Zhaojin Wen](https://www.linkedin.com/in/zhaojin-wen-7657bb220/),
[Andrew Li](https://www.linkedin.com/in/andrewli2403/),
[Jiantao Jiao](https://people.eecs.berkeley.edu/~jiantao),
and [Kannan Ramchandran](https://people.eecs.berkeley.edu/~kannanr/)

## Usage

Run the following to download the correctness data we used to train our model:
```sh
python download_data.py
```

To train a KNN and check its performance on correctness forecasting, simply do:
```sh
python knn.py
```

To run our Matrix Factorization model and see its performance, simply do:
```sh
python mf.py
```

For customized configuration, please see the argparse section in the code. 
