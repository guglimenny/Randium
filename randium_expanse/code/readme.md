# Code for making a Randium simulation

This folder contain code for the Randium model of viscous liquid dynamics.
The implimentation uses the Numba JIT for efficient calculations on the CPU and GPU. 

## Setup of Python

Before attempting to running the code, it is recommended to ensure that Numba for works. 
See Numba's documetation for more: [numba.readthedocs.io](https://numba.readthedocs.io)

Then type following to virtual enviroment:

    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

## Run the code with pure functions

The core for running a simulation is in the `backend.py` file.
The file can be executed as a script, to conduct a basic simulation:

    python3 randium_2d_gpu/backend.py

## Run the code using a front-end class

The file `randium_2d_gpu/randium_2d_gpu.py` contain a class using function in the `backend.py` file. As an usage example, see `run.py`

    python3 run.sh

The output of the simulation is located in the `Data` folder.
The result of an simulation is shown in the folder.
