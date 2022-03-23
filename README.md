# Multi-Level Attention: Sub-Instruction Understanding in Continuous Vision and Language Navigation
Official implrementations of *Multi-Level Attention: Sub-Instruction Understanding in Continuous Vision and Language Navigation* ([paper](https://arxiv.org/abs/2004.02857))

This repository is based on [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

## Setup
1. Use [anaconda](https://anaconda.org/) to create a Python 3.6 environment:
```bash
conda create -n vlnce python3.6
conda activate vlnce
```
2. Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7:
```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```
3. Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) 0.1.7:
```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```
4. Clone this repository and install python requirements:
```bash
git clone git@github.com:jacobkrantz/VLN-CE.git
cd MLA
pip install -r requirements.txt
```
5. Download Matterport3D sences:
   + Get the official `download_mp.py` from [Matterport3D project webpage](https://niessner.github.io/Matterport/)
   + Download scene data for Habitat
    ```bash
    # requires running with python 2.7
    python download_mp.py --task habitat -o data/scene_datasets/mp3d/
    ```
   + Extract such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.
6. Download preprocessed episodes [R2R_VLNCE_NRSub](https://github.com/RavenKiller/R2R_VLNCE_NRSub) from [here](https://drive.google.com/file/d/1rJn2cvhlQ7-GZ-gcUjJAjbyxfguiz2vv/view?usp=sharing). Extrach it into `data/datasets/`.
7. Download the depth encoder `gibson-2plus-resnet50.pth` from [here](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo). Extract the contents to `data/ddppo-models/{model}.pth`.

## Train, evaluate and test
`run.py` is the program entrance. You can run it like:
```bash
python run.py \
  --exp-config {config} \
  --run-type {type}
```
`{config}` should be replaced by a config file path; `{type}` should be `train`, `eval` or `inference`, meaning train models, evaluate models and test models.

Our config files is stored in `vlnce_baselines/config/mla`:
| File | Meaning |
| ---- | ---- |
| `mla.yaml` | Train base model |
| `mla_da.yaml` | Train model with DAgger |
| `mla_aug.yaml` | Train model with EnvDrop augmentation |
| `mla_da_aug_tune.yaml` | Fine-tune model with DAgger |
| `eval_single.yaml` | Evaluate and visualize a single path |




## Performance
The best model on validation sets is trained with EnvDrop augmentation and then fine-tuned with DAgger. We use the same strategy to train the model submitted to the test [leaderboard](https://eval.ai/web/challenges/challenge-page/719/leaderboard/1966), but on all available data (train, val_seen and val_unseen).
| Split      | TL   | NE   | OS   | SR   | SPL  |
|:----------:|:----:|:----:|:----:|:----:|:----:|
| Test       | 8.85 | 7.91 | 0.36 | 0.28 | 0.25 |
| Val Unseen | 8.27 | 7.60 | 0.36 | 0.29 | 0.27 |
| Val Seen   | 9.06 | 7.21 | 0.44 | 0.34 | 0.32 |

## Checkpoints
| Checkpoint | Link |
| ---- | ---- |
| Validation | Train base model |
| Test | Train model with DAgger |