## Getting Started
### Environment Requirements

This is the origin Pytorch implementation of the following paper: Rethinking and Improving Long-term Time Series Forecasting.


First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n LTSF python=3.8
conda activate LTSF
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example
- In `scripts/ `, we provide the model implementation *Dlinear/Autoformer/Informer/Transformer*
- In `FEDformer/scripts/`, we provide the *FEDformer* implementation
- In `Pyraformer/scripts/`, we provide the *Pyraformer* implementation

For example:

To train the **LTSF-Linear** on **Exchange-Rate dataset**, you can use the scipt `scripts/EXP-LongForecasting/Linear/exchange_rate.sh`:
```
sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh
```
It will start to train DLinear by default, the results will be shown in `logs/LongForecasting`. You can specify the name of the model in the script. (Linear, DLinear, NLinear)

All scripts about using LTSF-Linear on long forecasting task is in `scripts/EXP-LongForecasting/Linear/`, you can run them in a similar way. The default look-back window in scripts is 336, LTSF-Linear generally achieves better results with longer look-back window as dicussed in the paper. 

Scripts about look-back window size and long forecasting of FEDformer and Pyraformer is in `FEDformer/scripts` and `Pyraformer/scripts`, respectively. To run them, you need to first `cd FEDformer` or `cd Pyraformer`. Then, you can use sh to run them in a similar way. Logs will store in `logs/`.

Each experiment in `scripts/EXP-LongForecasting/Linear/` takes 5min-20min. For other Transformer scripts, since we put all related experiments in one script file, directly running them will take 8 hours-1 day. You can keep the experiments you interested in and comment out the others. 

## Acknowlegement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020
https://github.com/zhouhaoyi/ETDataset
https://github.com/laiguokun/multivariate-time-series-data
https://github.com/cure-lab/LTSF-Linear
