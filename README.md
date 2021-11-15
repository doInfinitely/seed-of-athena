In order to use call

```
python suggest.py
```

If the initial theta is omitted, then the program assumes the initial theta to
be the median of the thetas computed during training. The program outputs an
updated theta value, and suggested question ids (naturally, suggestions will
not include already asked questions).

This code is intended to be used in an online iterative fashion, 1) first call
suggest.py with a blank input file (or without an input file) to get suggested
question ids. 2) Update the initial theta in the input file with the output of
suggest.py 3) Present one of the suggested questions to the examinee and add
its question id to either the list of correct or incorrect questions.

This online procedure should be halted when abs(updated theta - initial theta) 
< delta, where delta is at the discretion of the tester. Generally, smaller
values of delta will lead to longer adaptive testing sessions.

Example usage:

```
>> python suggest.py
>> -0.78297361052415 [3584]
>> python suggest.py -0.78297361052415 --correct 3584
>> 16.419705703855662 [5333]
>> python suggest.py -0.78297361052415 --correct 3584 --incorrect 5333
>> 0.8964191311240539 [3534]
>> python suggest.py -0.78297361052415 --correct 3584 5880 3534 --incorrect 5333
>> 1.1473318583024894 [5978]
```

Update 09/20/20:

Did some hype-parameter tuning, fleshed out the code to run simulated computer
adaptive tests on the training set.

Added code to output theta values by percentile.

Example to get a cutoff theta value such that 15% of students are below the
cutoff:
```
>> python get_theta_by_percentile.py 15
```

Update 10/21/20

Made some quality of life improvements to the training, added checkpointing so
that training can be stopped and resumed and also added detection for when the
model has converged on the optimum. The code checks for the 'checkpoint.p'
file, delete this file in order to train from scratch.

Update 11/05/20

Added gpu acceleration and made other optimizations to dramatically speed up
training. Added a section at the top of the train.py and the suggest.py files
for hyper-parameters.
