# OMsignal


## Directory Structure
* `src` Main directory for source code.
* `src/algorithm`: class definitions for models used
* `src/legacy`: code from block 1 teams, included as is
* `src/scripts`: runnable scripts that calls the class definitions and train models
* `src/utils`: utility functions, e.g. saving models, tensorboard plotting, constants and file paths 
* `data` Data related utilities. Data Loaders. **NOTE: DO NOT UPLOAD ANY REAL DATA.**
* `model` Binary model saved after training
 


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

## Coding style auto-refactor tool

after installing virtual env on conda, do 
```
source activate calqc

pip install black
```

Now, add Black to your Pycharm, see the official instructions [here](https://pypi.org/project/black/)


## Project mangement

### Important dates
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






