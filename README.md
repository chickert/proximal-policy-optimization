
# Implementation of proximal policy optimization algorithm

## How to run
Run the reacher_training.py, reacher_wall_training.py and pusher_training.py scripts in the experiments folder. 

## Requirements
```
-e git+https://github.com/Improbable-AI/airobot.git@88892832289c9b129375cc944e7093a4da9e07d5#egg=airobot
torch==1.4.0
numpy==1.18.1
```

## Versioning
The master version of this repo accomodates discrete action spaces only. The 'continuous' branch has an implementation that allows for continuous action spaces as well. 
