# FuzzyFormer: A Fuzzy Attention-Enhanced Transformer for Long-Term Time Series Forecasting

1\. Environment Setup
---------------------

We recommend using **conda** to create a clean Python environment (tested under **Python 3.8**):

```bash
conda create -n fuzzyformer python=3.8 -y
conda activate fuzzyformer
```

* * *

2\. Installation
----------------

After activating the environment, install the dependencies with:

```bash
pip install -r requirements.txt
```

* * *

3\. Data Preparation
--------------------

You can directly download the **pre-processed datasets** used in our paper from:

*   [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link)
*   [Baidu Netdisk](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry)

Then place the downloaded data under the folder `./dataset`.

The folder structure should look like:

```
dataset/
  ├── ILI.csv
  ├── ETTh1.csv
  ├── ETTm2.csv
  └── ...
```

* * *

4\. Train & Evaluate
--------------------

We provide **experiment scripts** for all benchmarks in `./scripts/multivariate_forecast`

For example, to reproduce the **ILI dataset** experiment with FuzzyFormer, simply run:

```bash
sh ./scripts/multivariate_forecast/ILI_script/FuzzyFormer.sh
```

* * *

Acknowledgement
---------------

This library is constructed based on the following repos:

*   [https://github.com/decisionintelligence/TFB](https://github.com/decisionintelligence/TFB)
*   [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

* * *