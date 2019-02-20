# OMsignal

## Use virtual env with conda
Create your Anaconda environment.

```
conda create -n [your fav name for virtual env] python=3.6
```
Here I use the env name `calqc` as an example

Next, activate your environment.

```
source activate calqc
```

Install Python dependencies. I did a pip freeze on Helios, install the following locally to make sure that you can the same setup as helios. I've had the issue where Helios only recognize numpt 1.16 but I had something else. Will not re-live that nightware ever again

```
pip install -r requirements.txt
```

## IDE: Run and debug remotely with Pycharm (without Jupyter)
I strongly recommend pycharm. Here's an excerpt of my block 1 team's setup.


### Requirements
* Pycharm **Professional edition** 2018.3.3 (works with 2018.2.7 too)
    * The professionnal edition is free (and easy to get) for students as long as you give an university email :  https://www.jetbrains.com/student/ 
### Setup
1. Run this ssh tunnel command in a command line: `ssh -N -L 4444:localhost:22 USERNAME@helios.calculquebec.ca -p 22`. Now, the "remote" IP will be localhost and the port will be 4444.
Without this tunnel, the remote interpreter python console will not connect.
See more here : https://youtrack.jetbrains.com/issue/PY-18029 and here : https://www.ssh.com/ssh/tunneling/
2. In Pycharm, you need to setup the Deployment configuration in the Settings menu. Add a new configuration and
choose the SFTP option. Add your username, your password. Make sure that the remote address is localhost and the port is 4444 (because of the ssh tunnel).
3. Now, in the same window, in the **Mappings** tab, you need to specify the **Deployment** path. It will copy your local project to the desired path in 
your home directory in helios (for me, it is :  /DeployedProject/horoma). You can then enable automatic synchronization with Tool->Deployment->Automatic upload (really useful).
4. To upload your project for the first time, you then need to right click your local root project folder (ift6759-horoma) and Deployment-> Upload to x.
5. In Pycharm, you now need to setup the remote interpreter. To do this, go to File->Settings->Project Interpreter and add a new remote interpreter.
Select the deployment that you have just made.
6. **Really Important** : You need to change the default path to the python interpreter, because the real interpreter is in a singularity container (the same container that you can reach when you execute the s_shell command).
To link it, you need to set the path to a shell script that I've written (ex: `/home/USERNAME/ETC/singularity_python.sh`). What this script does is like the s_shell command, but instead of running a shell in the container, it directly execute python  in the container.
7. **Really Important 2** : if it does not work, you may need to give the right access to this file with this command : `chmod 755 /home/USERNAME/ETC/singularity_python.sh`
8. Now, you can try to run a test python program to see if it works!
9. **Optional**: I recommend to set the **project** interpreter to a local one and only use the remote one when you run a python file (in the run configuration tab).
 I think that it is not really worth it to set the project interpreter to the remote one because it needs a lot of pycharm processing. If you do that, you need to make sure that you have the same python packages as the remote python interpreter ones.

## Coding style auto-refactor tool

after installing virtual env on conda, do 
```
source activate calqc

pip install black
```

Now, add Black to your Pycharm, see the official instructions [here](https://pypi.org/project/black/)


# Project mangement

## Important dates
 * report + code + best model 2019/03/15 23:59 

## Github proper usage
### Proper Commit message
```

How To Write Proper Git Commit Messages
Go to the profile of Stephen Amaza
Stephen Amaza
Mar 29, 2017

A git commit records changes to a repository.

A git commit is executed in the course of a project to record progress. This progress is then pushed to a remote repository (like on github.com) by executing a git push. These changes or progress could mean any additions, deletions, or updates to the files stored in the repository.

Overview of a Proper Git Commit

Credit: Imgur
The quickest way to write a git commit is to use the command git commit -m "Git commit message here". This is not recommended for commits, however, because it provides limited description of what was changed. Essentially, a git commit should explain what and why a change has been made.

Here is the guideline for writing a full git commit.

A subject which contains the title of the git commit. This should be limited to 50 characters and should be separated from the rest of the commit with a blank line.
Explanatory text explaining what has been changed and why the change was necessary. Write in the imperative mood e.g. “Fix bug causing outage” rather than “Fixed bug causing outage”.
See below for an example of an implementation of the above.

Capitalized, short (50 chars or less) summary
More detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.
Write your commit message in the imperative: "Fix bug" and not "Fixed bug"
 or "Fixes bug."  This convention matches up with commit messages generated
by commands like git merge and git revert.
Further paragraphs come after blank lines.
- Bullet points are okay, too
- Typically a hyphen or asterisk is used for the bullet, followed by a single space, with blank lines in between, but conventions vary here
- Use a hanging indent
```
[Source](https://medium.com/@steveamaza/how-to-write-a-proper-git-commit-message-e028865e5791)


# prelim to-dos
- [ ] in corporate past student code + TA baseline code
- [ ] save the log files and bring it offline to plot - > automate this in shell script including the scp
- [ ] email notif for log jobs on helios



## TA's recommended timeline

### 2019/02/11 week
Review code and reports from block 1
Understand how to use TensorboardX
Code data loader for unlabeled data


### 2019/02/18 week
Identify and start implementing multi-task solutions for incorporating unlabeled data into the training process


### 2019/02/25 week
Continue implementing multi-task solution, leveraging the unlabeled data
Hyper parameter fine tuning

### 2019/03/01 week
Write a short report summarizing the work, and results
Provide model for blind test set evaluation
Complete the peer code review



