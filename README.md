# Project1: Navigation ( Deep RL NanoDegree )

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. 

## Project Description

Implement a Deep Q Network - DQN - to solve the task provided. 

Task:
- Train an agent to navigate the given Unity environment, while maximizing the output score.
- Collect as many yellow bananas as possible
- Avoid collecting the blue bananas along the way

## Environment

Unity Banana Environment - Used to train an agent to learn to navigate the environment, while collecting yellow bananas along the way. For more information please refer to project introduction found here: [Starter project files](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

## Getting Started 

Note the following conda environment was created on a Linux machine ( __Linux Ubuntu 20.04.6 LTS__ ) 

1. Please clone the following Udacity Repo: [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) Repo.

2. Follow the instructions - inside the repo - to set up the necessary dependencies. 
* Create (and activate) a new conda environment with Python 3.6 : 
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
* Install OpenAI gym
```bash
pip install gym
```
* Install the necessary Python dependencies 
```bash
cd deep-reinforcement-learning/python
```
- Modified Requirements.txt 
```text
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==1.4.0
pandas
scipy
ipykernel
```
```bash
pip install .
```
* Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
* Download the Unity Banana Environment and place the file ( decompress ) inside this repo :
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
After configuration, please refer to the jupyter notebook: Navigation.ipynb
