# TV-Script-Generation

## Table of Contents

1. [Introduction](#introduction)
2. [Directory Structure](#directoryStructure)
3. [Installation](#installation)
4. [Instructions](#instructions)
5. [Results](#results)

## Directory Structure <a name="directoryStructure"></a>

- Root /
    - README.md (This readme file)
    - data_loader.py
    - dlnd_tv_script_generation.ipynb
    - helper.py
    - preprocess.p (persisted preprocessed data)
    - preprocessing.py
    - problem_unittests.py
    - rnn.py
    - workspace_utils
    - data /  
        - Seinfeld_Scripts.
    - generated_scripts /  
        - generated_script_george_4.txt
        - generated_script_jerry_4.txt
        - generated_script_jerry_5.txt
        - generated_script_kramer_4.txt
        - generated_script_kramer_5.txt
        
## Installation <a name="installation"></a>

This project was written in Python 3.6, using a Jupyter Notebook on Anaconda. Currently (September 2019) you cannot use Python 3.7, since tensorflow 1.7.1 doesn't have a version corresponding to python 3.7 yet.

The relevant Python packages for this project are as follows:
 - os
 - pickle
 - signal
 - numpy
 - torch
 - torch.utils.data
 - torch.nn
 - unittest.mock

    