# DDDQN-Bird

Flappy Bird with a Double Dueling Deep Q-Network

Game adapted from: https://github.com/sourabhv/FlapPyBird

## Quickstart

All requirements are saved in environment.yml  

* install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* cd into the root directory of this repository
* create a new environment from the environment.yml file
```
conda env create -f environment.yml
```

* activate the new environment
```
conda activate rl

# on fedora
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
```

## Synopsis

### Play normal flappy bird
```
python dddqn-flappy.py play
```

### Train a model
```
python dddqn-flappy.py train --headless --episodes 100000
```

### Evaluate a trained model
A model trained for 75000 simulations is saved in ```checkpoints/```


```
python dddqn-flappy.py evaluate PATH/TO/TRAINED/MODEL
```
