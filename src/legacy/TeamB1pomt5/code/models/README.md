# Models

This directory contains several trained models.

The following models are trained with the optimal hyperparameters on only the supplied training set:

* `final_report_PR_CNN`, which trains the `PR_CNN`
* `final_report_RT_Ranker`, which trains the `RT_Ranker`
* `final_report_ID_CNN`, which trains the `ID_CNN`

The following models are trained with the optimal hyperparameter on the combined training and validation sets (to be used against the unseen test set):

* `final_eval_PR_CNN`, which trains the `PR_CNN`
* `final_eval_RT_Ranker`, which trains the `RT_Ranker`
* `final_eval_ID_CNN`, which trains the `ID_CNN`

Finally, `rr_stdev.model` is used to train the `RR-Regressor`.
