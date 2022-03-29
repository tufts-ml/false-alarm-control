

# Optimizing Early Warning Classifiers to Control False Alarms Via a Minimum Precision Constraint

This repo contains code for training binary classifiers to maximize recall subject to a minimum precision constraint.

We have proposed a new method, formally described in [Rath & Hughes AISTATS 2022](https://github.com/tufts-ml/false-alarm-control/#citation).

The focus of this repo is currently a *toy* binary classification task described in our paper. This focus makes comparing different techniques simple and visually comparing their results easy.

We provide a notebook comparing 4 different methodologies:
* BCE + threshold search
* Eban et al's hinge bound
* Fathony & Kolter's adversarial prediction bound
* Our proposed sigmoid bound

All methods try to maximize recall subject to satisfying a precision constraint.

### Workflow

 1. Users are expected to installing the conda enviroment via the [toy_false_alarm_control.yml](toy_false_alarm_control.yml) file provided

```python
>> conda env create --name toy_false_alarm_control --file=toy_false_alarm_control.yml
>> conda activate toy_false_alarm_control
```

2. Open the [notebook for reproducing and comparing multiple bounds on the toy example](toy_example_comparing_BCE_Hinge_and_Sigmoid.ipynb) 

3. Run the cells to create the toy example. Our toy example is heavily imbalanced with 120 positive examples and 450 negative examples.

![](images/toy_example.png?raw=true)

 4. Our goal is to train a linear classifier to find a decision boundary that maximizes recall subject to precision>=0.9.

   - **BCE + Threshold search :** We first try Binary cross entropy + post-hoc threshold search, which is commonly used in many applications for the meeting the desired precision-recall. 

![](images/BCE_plus_threshold_search_solution.png?raw=true)
    
   However, we see that even with post-hoc search, BCE cannot achieve the desired precision.

   - **Eban et al's hinge bound :** We then try [Eban et al's](http://proceedings.mlr.press/v54/eban17a/eban17a.pdf) proposed hinge bound.  
   
   
![](images/hinge_solution_precision_90.png?raw=true)


   Here again, the hinge bound falls short of the desired precision of 0.9, reaching 0.79 instead. We hypothesize that this is due to the looseness of the hinge bound.

   - **Fathony & Kolter's adversarial prediction bound :** Next, we try the optimizing our custom objective using an adversarial prediction framework recent proposed by [Fathony & Kolter](http://proceedings.mlr.press/v108/fathony20a.html).

![](images/adversarial_prediction_precision_90.png?raw=true)

   The adversarial prediction bound reached the desired precision of 0.9, and is able to achieve a recall of 0.11, without any post-hoc threshold search. However the total runtime is nearly 3000 seconds, which is 300x the training time required for the other 3 methods.

   - **Our proposed sigmoid bound :** Finally, we show the decision boundary of our proposed sigmoid bound, which is tight, differentiable, making gradient-based learning feasible.
   
![](images/sigmoid_solution_precision_90.png?raw=true)

   Our proposed sigmoid bound reaches the desired precision of 0.9, without any post-hoc threshold search, and achieves a recall of 0.23, which is nearly 2x the recall achieved by Fathony & Kolter's adversarial prediction bound. Moreover our proposed bound requires a training time of ~15 seconds, which is $(1/300)^{th}$ of the training time required by Fathony & Kolter's adversarial prediction bound.


## Experiments on EHR Data
We will be providing polished code for experiments on the EHR data as detailed in [Rath & Hughes AISTATS 2022](https://github.com/tufts-ml/false-alarm-control/#citation) by the end of April 2022. Users can replicate the results by running snakemake files in the following format :

### Standardizing the dataset
```python
>> snakemake --cores 1 --snakefile standardize_dataset_and_split_train_test.smk 
```

### Training the logistic regression and neural network with multiple training objectives written with pytorch
```python
>> snakemake --cores 1 --snakefile train_{model}.smk BCE_plus_threshold_search
>> snakemake --cores 1 --snakefile train_{model}.smk hinge_bound
>> snakemake --cores 1 --snakefile train_{model}.smk sigmoid_bound
```

### Evaluating the performance of trained models
```python
>> snakemake --cores 1 --evaluate_performance.smk
```

### Expected output
![](images/model_comparison.png?raw=true)

## Citation

If you use this code, please cite our manuscript, published in the proceedings of AISTATS 2022.

<blockquote>
<p>
<i>Optimizing Early Warning Classifiers to Control False Alarms via a Minimum Precision Constraint</i>.
 <br />
Preetish Rath and Michael C. Hughes
 <br />
In Proceedings of Artificial Intelligence and Statistics (AISTATS), 2022.
 <br />
PDF available: <a href="https://www.michaelchughes.com/papers/RathHughes_AISTATS_2022.pdf">https://www.michaelchughes.com/papers/RathHughes_AISTATS_2022.pdf</a>
</p>
</blockquote>
    

```
@inproceedings{rathOptimizingEarlyWarning2022,
    title = {Optimizing Early Warning Classifiers to Control False Alarms via a Minimum Precision Constraint},
    booktitle = {Artificial Intelligence and Statistics (AISTATS)},
    author = {Rath, Preetish and Hughes, Michael C.},
    year = {2022},
    url = {https://www.michaelchughes.com/papers/RathHughes_AISTATS_2022.pdf},
}
```
