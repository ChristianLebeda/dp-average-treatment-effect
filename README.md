# Model Agnostic Differentially Private Causal Inference

This folder contains the implementation for the experiments in the paper: Model Agnostic Differentially Private Causal Inference. 

Note that we do not include the baselines from prior work as we used source code which we do not have the rights to distribute.

We also do not include the experiment with real data. We cannot share access to the dataset due to the sensitive nature of the data.  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Generating Plots

To create the plots from the paper run the following commands from the plots folder

```setup
python3 figure1.py
python3 figure2.py
python3 figure3.py
python3 figures4_5.py
python3 figure6.py
```

Note that the figure numbers differ from the current version of the paper. 

## Running Experiments

Use the run_experimentX.py scripts to recreate an experiment from the paper. E.g. to run the experiment with low overlap use

```setup
python3 run_experiment1.py
```

This will output the estimated ATE for well-specified G-formula and AIPW before Gaussian noise is added. Since we add noise as the final step of our mechanism, we can add noise when plotting the data.

Parameters such as "K" can be updated by modified the scripts.

## Note

Running the experiments will result in slightly different results for some of the experiments. 
Due to some clean-up of the source code the random seed for data generation does not match that used at the submission deadline.
This slight discrepency obviously does not change any conclusions of the paper.
Nonetheless, the figures will be updated for a future version for improved reproducibility.

## Explanation of different experiments

The script order does not match one-to-one with the figures in the paper.

run_experiment1.py contains the experiment with low overlap depicted in Figure 1 (left).  
run_experiment2.py contains the misspecified setting with tree models depicted in Figure 1 (right).  
run_experiment3.py contains the well-specificied setting with good overlap depicted in Figure 3.  
run_experiment4.py examines the effect of changing the parameter K depicted in Figures 5 and 6.  
run_experiment5.py contains the comparison between non-private and private G-Formula in Figure 2.
run_experiment_SA.py uses the subsample and aggregate framework, the results are depicted in Figure 7.
