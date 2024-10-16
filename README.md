# TD-MPC in tinygrad
This repository contains an implementation of [Temporal Difference Model Predictive Control](https://www.nicklashansen.com/td-mpc/) using tinygrad. As a warning, it's in a *really* rough condition when it comes to the training code. Disregard for now.
## Overview
TD-MPC is a model-based reinforcement learning algorithm that performs local trajectory optimization in the latent space of a learned implicit world model. This implementation uses tinygrad, a lightweight deep learning framework.
## Features
Implicit world model implementation
MPC-based planning in latent space
Temporal difference learning for value estimation
Support for continuous control tasks
## Installation
```
pip install tinygrad numpy
```
## Usage
```
METAL=1 BEAM=2 DEBUG=2 python3.12 main_train.py
```
