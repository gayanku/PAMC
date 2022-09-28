# Proxy approximated meta-node Contrastive (PAMC) loss
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a PyTorch implementation of "Efficient block contrastive learning via parameter-free meta-node approximation"

TLDR; A simple block contrastive loss approximation technique to efficiently contrast all negative samples, in linear cluster time, at graph level

TIn a nutshell : USPS image dataset : ***CUDA training time 16.117s with PAMC Approximation, 50.388s without!. 3.12 times faster!!!*** 

The repo, inlcuding data sets and pretrained models are, has been forked initially from [SCGC](https://github.com/gayanku/SCGC) which uses portions of code from [SDCN](https://github.com/bdy9527/SDCN), [AGCN](https://github.com/ZhihaoPENG-CityU/MM21---AGCN) and [Graph-MLP](https://github.com/yanghu819/Graph-MLP). 

## Setup
- Our code was tested on CUDA 11.3.0, python 3.6.9, pytorch 1.3.1.
- `pip install -q munkres` is needed for the hungarian algorithem, for the evaluation metrics
- The code also run on Google colab (2022 April) with no modifications, with munkres installed.

Note : The code can be tested with no GPU if the GPU timing code is commented out. 

## Datasets

The dataset contains 2 folders, `data` and `graph`. Please obtain them from the [dataset Google drive links](https://github.com/bdy9527/SDCN/blob/master/README.md). You will need to set `--data_path` to the parent folder containing `data` and `graph`. Please note that the `data` folder contains the pre-trained `.pkl` models. We directly use these pre-trained models from SDCN.


## Usage
- All parameters are defined in train.py with comments and explanations. 

- To run using PAMC on the 6 datasets, for 10 iterations, use the following. This code has GPU time and memory profiling enabled, which can be turned off by commenting relevant code. Our published ACC,NMI,ARI and F1 was run with profiling commented. 
```
$SCRIPT --name usps --iterations 10 --epochs 200 --model SCGC_TRIMPOSCCSP --verbosity 0   --alpha 2    --beta 2   --order 4 --tau 0.5    --lr 0.001   --influence 
$SCRIPT --name hhar --iterations 10 --epochs 200 --model SCGC_TRIMPOSCCSP --verbosity 0   --alpha 0.5  --beta 12.5  --order 2 --tau 1.5  --lr 0.001  --influence    
$SCRIPT --name reut --iterations 10 --epochs 200 --model SCGC_TRIMPOSCCSP --verbosity 0   --alpha 1  --beta 0.2 --order 1 --tau 0.25   --lr 0.0001  --influence 
$SCRIPT --name acm  --iterations 10 --epochs 200 --model SCGC_TRIMPOSCCSP --verbosity 0   --alpha 0.5   --beta 0.5   --order 1 --tau 0.5   --lr 0.001   --influence 
$SCRIPT --name dblp --iterations 10 --epochs 200 --model SCGC_TRIMPOSCCSP --verbosity 0   --alpha 2    --beta 2   --order 1 --tau 1      --lr 0.001   --influence  
$SCRIPT --name cite --iterations 10 --epochs 200 --model SCGC_TRIMPOSCCSP --verbosity 0   --alpha 2    --beta 2.5 --order 3 --tau 0.5    --lr 0.001   --influence  
```
The usps output is
```
Namespace(alpha=2.0, batch_size=2048, beta=2.0, cuda=True, data_path='/content/drive/MyDrive/001_Clustering/_Dataset_SDCN', epochs=200, influence=True, iterations=10, k=3, lr=0.001, mode='full', model='SCGC_TRIMPOSCCSP', n_clusters=10, n_input=256, n_z=10, name='usps', note='-', order=4, seed=42, tau=0.5, verbosity=0)
---------------PROFILING CODE--------------
Loaded PAE acc:0.7098  nmi:0.6748  ari:0.5874  f1:0.6968
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total GFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       _MODEL_TRAIN_ALL        67.07%       40.307s        98.49%       59.188s       59.188s       0.000us         0.00%        6.680s        6.680s          -4 b    -253.44 Mb           0 b     -85.15 Gb             1            --  
                                           _MODEL_TRAIN         0.10%      60.000ms         0.41%     247.357ms       1.237ms       0.000us         0.00%        2.261s      11.304ms       2.34 Kb     -33.23 Kb     823.89 Mb     -42.81 Gb           200            --  
                                              _MODEL_KL         0.02%      11.559ms         0.11%      66.814ms     334.070us       0.000us         0.00%      13.703ms      68.515us         800 b     -20.62 Kb      71.00 Mb    -199.50 Kb           200            --  
                                            _MODEL_DIST         0.06%      38.008ms         3.59%        2.158s      10.788ms       0.000us         0.00%      28.992ms     144.960us        -800 b     -46.94 Kb      53.04 Mb    -277.99 Mb           200            --  
                                     _MODEL_CONTRASTIVE         0.11%      65.021ms         3.19%        1.916s       9.582ms       0.000us         0.00%        1.916s       9.581ms       2.34 Kb      -7.11 Mb      40.02 Gb     -53.25 Gb           200            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 60.093s
Self CUDA time total: 16.117s
Z:acc-nmi-ari-F1-gpu-clock:  0.8491,  0.8407,  0.8420,  0.0024,|, 0.8146,  0.8016,  0.8031,  0.0038,|, 0.7945,  0.7753,  0.7775,  0.0057,|, 0.7933,  0.7872,  0.7882,  0.0017,|,53943.8125, 50873.8398, 52022.0410, 957.9162,|,53.9433, 50.8735, 52.0215,  0.9579,||, Namespace(alpha=2.0, batch_size=2048, beta=2.0, cuda=True, data_path='/content/drive/MyDrive/001_Clustering/_Dataset_SDCN', epochs=200, influence=True, iterations=10, k=3, lr=0.001, mode='full', model='SCGC_TRIMPOSCCSP', n_clusters=10, n_input=256, n_z=10, name='usps', note='-', order=4, seed=42, tau=0.5, verbosity=0)
```
The line `Z:acc-nmi-ari-F1-gpu-clock` gives the min,max,avg,std of ACC,NMI,ARI,F1,GPU,CPU followed by || and all the args.
Profiling `_MODEL_XXX` contexts capture logical model functions and training. Please see the code for more information.


- To see how the model would train without PAMC approximations, run SCGC*, as follows. 
```
python train.py --name usps --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 4 --beta 0.1 --order 4 --tau 0.25 --lr 0.001 --influence
python train.py --name hhar --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 10  --order 3 --tau 2.25 --lr 0.001 --influence
python train.py --name reut --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 0.5 --beta 0.1 --order 3 --tau 0.25 --lr 0.001 --influence
python train.py --name acm  --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.001 --influence
python train.py --name dblp --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.001 --influence
python train.py --name cite --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.0001 --influence
```
The usps output is
```
Namespace(alpha=4.0, batch_size=2048, beta=0.1, cuda=True, data_path='/content/drive/MyDrive/001_Clustering/_Dataset_SDCN', epochs=200, influence=True, iterations=1, k=3, lr=0.001, mode='full', model='SCGC_TRIM', n_clusters=10, n_input=256, n_z=10, name='usps', note='-', order=4, seed=42, tau=0.25, verbosity=0)
---------------PROFILING CODE--------------
Loaded PAE acc:0.7098  nmi:0.6748  ari:0.5874  f1:0.6968
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total GFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       _MODEL_TRAIN_ALL        27.73%       50.714s        99.25%      181.531s      181.531s       0.000us         0.00%       26.670s       26.670s          -4 b      -2.41 Gb           0 b    -367.99 Gb             1            --  
                                           _MODEL_TRAIN         0.06%     109.462ms         0.23%     421.977ms       2.110ms       0.000us         0.00%        3.386s      16.931ms       2.34 Kb        -800 b     876.65 Mb     -42.76 Gb           200            --  
                                              _MODEL_KL         0.01%      19.598ms         0.07%     118.949ms     594.745us       0.000us         0.00%      11.177ms      55.885us         800 b      -2.34 Kb      71.00 Mb    -199.50 Kb           200            --  
                                            _MODEL_DIST         1.03%        1.888s        14.81%       27.096s     135.478ms       0.000us         0.00%       15.566s      77.831ms        -800 b     -64.41 Gb     258.14 Gb    -193.11 Gb           200            --  
                                     _MODEL_CONTRASTIVE         0.03%      48.284ms         0.08%     149.847ms     749.235us       0.000us         0.00%        4.008s      20.040ms         800 b      -2.34 Kb      64.48 Gb    -128.92 Gb           200            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 182.898s
Self CUDA time total: 50.388s
Z:acc-nmi-ari-F1-gpu-clock:  0.8489,  0.8489,  0.8489,  0.0000,|, 0.8411,  0.8411,  0.8411,  0.0000,|, 0.7941,  0.7941,  0.7941,  0.0000,|, 0.8152,  0.8152,  0.8152,  0.0000,|,106441.4219, 106441.4219, 106441.4219,  0.0000,|,106.4405, 106.4405, 106.4405,  0.0000,||, Namespace(alpha=4.0, batch_size=2048, beta=0.1, cuda=True, data_path='/content/drive/MyDrive/001_Clustering/_Dataset_SDCN', epochs=200, influence=True, iterations=1, k=3, lr=0.001, mode='full', model='SCGC_TRIM', n_clusters=10, n_input=256, n_z=10, name='usps', note='-', order=4, seed=42, tau=0.25, verbosity=0)
Namespace(alpha=1.0, batch_size=2048, beta=10.0, cuda=True, data_path='/content/drive/MyDrive/001_Clustering/_Dataset_SDCN', epochs=200, influence=True, iterations=1, k=5, lr=0.001, mode='full', model='SCGC_TRIM', n_clusters=6, n_input=561, n_z=10, name='hhar', note='-', order=3, seed=42, tau=2.25, verbosity=0)
```

- Note the CUDA time : 16.117 with PAMC, 50.388 without!. An increase of 3.12 times!!!


## Data sources and code
Datasets and code is forked from [SDCN](https://github.com/bdy9527/SDCN). We use also use the model code from AGCN [AGCN](https://github.com/ZhihaoPENG-CityU/MM21---AGCN) and portions of contrastive loss code from [Graph-MLP](https://github.com/yanghu819/Graph-MLP). We acknowledge and thank the authors of these works for sharing their code.

## Citation
```
ANON
```
