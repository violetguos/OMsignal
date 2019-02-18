# OMsignal

# set up

# Install

Create your Anaconda environment.

```
conda create -n calqc python=3.6
```

Next, activate your environment.

```
source activate calqc
```

Install Python dependencies.

```
pip install -r requirements.txt
```




# technical util
* save the log files and bring it offline to plot - > automate this in shell script including the scp

2019/02/11 week
2019/02/18 week
2019/02/25 week
2019/03/11 week
●Review code and reports from block 1
●Understand how to use TensorboardX
●Code data loader for unlabeled data
●Identify and start implementing multi-task solutions for incorporating unlabeled data into the training process
● Continue implementing multi-task solution, leveraging the unlabeled data
●Hyper parameter fine tuning
●Write a short report summarizing the work, and results
●(Peer-) Review of other teams' code
●Have a clear understanding of the data
●Data loader for unlabeled data
●Understanding of solutions for incorporating unlabeled data into the training process
●Multi-task model with unlabeled data (beginning of week after spring break)
●Produce documented code and report summarizing the experimental work
●Provide model for blind test set evaluation
●Complete the peer code review


Semi-Supervised Learning
● Class 1: a branch is added in the network for handling unlabeled data. (slide 51)
● Class 2:
○ Leave the model unchanged as for the fully
 supervised setting. (slide 52)
 Entropy Based Approaches
○ A simple loss term for the unlabeled data is added to encourage the network to make “confident” (low-entropy) predictions for all examples, regardless of the actual class predicted.
○ Pseudo Labeling
○ Consistency regularization

