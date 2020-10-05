# Data Poisoning Attacks on Regression Learning and Corresponding Defenses
This repository containts the code for the paper 'Data Poisoning Attacks on Regression Learning and Corresponding Defenses'
(see [ArXiv.org for full paper](https://arxiv.org/pdf/2009.07008.pdf)).
This repository implements the 'Flip' data poisoning attack on regression learning, and the
'Trim' and 'iTrim' defense. It evaluates these on 26 data sets.

### How to run:
1) Download and include the required data sets (see section `Datasets` below).
2) Set the python path of main.py to the base of this directory.
3) Set `settings.no_attack_run = True` and run `main.py`.
This will create the training data and split accordingly.
4) Set `settings.no_attack_run = False` and run `main.py`.
The code will parallelize all experiments over the available combinations of data
sets / regressors and finally output the results to the console.

### Datasets:
Unfortunately, we're not able to prodive the data sets used
in the repository (due to potential copyright issues).
Thus, we list what data sets we used and where to obtain them:

In the folder `res`, place the following data sets:

- Github_regression: https://github.com/paobranco/Imbalanced-Regression-DataSets
- Keel Data sets: https://sci2s.ugr.es/keel/datasets.php
- Three data sets provided by [1]: Warfarin [2], loan and armesHousing.

The file `res/tree.txt` provides the layout of the folder.


### Citation
If you use this work, please cite us:

Müller, Nicolas Michael, Daniel Kowatsch and Konstantin Böttinger. Data Poisoning Attacks on Regression Learning and Corresponding Defenses - IEEE Pacific Rim International Symposium on Dependable Computing (PRDC) 2020

### Bugs
If you find bugs, feel free to fix them and send us a pull request.

### References

[1] Jagielski,   M.,   Oprea,   A.,   Biggio,   B.,   Liu,   C.,   Nita-Rotaru,  C.,  and Li,  B.   Manipulating Machine Learn-ing:  Poisoning  Attacks  and  Countermeasures  for  Re-gression  Learning.InProceedings  -  IEEE  Sympo-sium  on  Security  and  Privacy,   volume  2018-May,pp.  19–35.  IEEE,  may  2018.   ISBN  9781538643525.doi:/10.1109/SP.2018.00057.URL/https://ieeexplore.ieee.org/document/8418594/.

[2] IWPC Data set: https://www.pharmgkb.org/downloads
