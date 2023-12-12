# BackProp Baseline
This repo is for experimenting with simple architectures using backpropagation to serve as baseline for my study on Forward Forward.

## How to Use

### Setup
- Install [conda](https://www.anaconda.com/products/distribution)
- Adjust the ```setup_conda_env.sh``` script to your needs (e.g. by setting the right CUDA version)
- Run the setup script:
```bash
bash setup_conda_env.sh
```


### Run Experiments
- Run the training and evaluation with forward-forward:
```bash
source activate FF
python -m main
```
