# TODO:
# USE the comet.ml for hyperparameter tuning, auto logging experiment
# run the test experiment with dummy data and basic basic AE

from comet_ml import Experiment
import os
import torch
import sys
import configparser
import argparse

from src.utils.os_helper import get_hyperparameters



"""
Uses external API comet to automatically log hyper param experiments,
 stored privately and approved by TA
"""

# TODO: TRY learning rate scheduler

def train_experiment(experiment, hyper_params, train_loader, model, optimizer, criterion, train_dataset):
    with experiment.train():
        for epoch in range(hyper_params['num_epochs']):
            correct = 0
            total = 0
            experiment.log_current_epoch(epoch)
            for i, (images, labels) in enumerate(train_loader):

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Compute train accuracy
                # TODO: Import the TA evaluation criteria for accuracy

                # Log to Comet.ml
                experiment.log_metric("loss", loss.data[0], step=i)
                experiment.log_metric("accuracy", correct / total, step=i)

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                          % (epoch + 1, hyper_params['num_epochs'], i + 1,
                             len(train_dataset) // hyper_params['batch_size'], loss.data[0]))


def test_experiment(experiment, test_loader, model):
    with experiment.test():
        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # TODO: change this to TA criteria
            correct += (predicted == labels).sum()

        experiment.log_metric("accuracy", 100 * correct / total)
        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


def run_experiment(arguments, hyper_parameters, config_file_name):
    experiment = Experiment(api_key=os.environ["COMET_API_KEY"],
                            workspace=os.environ["COMET_WORKSPACE"],
                            project_name=os.environ["COMET_PROJECT"])

    experiment.add_tag(arguments.tag_exp)
    experiment.add_tag(config_file_name)
    experiment_key = experiment.get_key()
    # TODO: use experiment key for the _suffix
    experiment.log_parameters(hyper_parameters)

    # model.run()

    # record the losses

    # experiment.log_metrics(test_metrics)
    # experiment.log_metrics(train_metrics)


def main(config_ae):
    # TODO: parse a config file

    # reads in the config files
    autoencoder_config = configparser.ConfigParser()
    autoencoder_config.read(config_ae)
    autoencoder_hp_dict = get_hyperparameters(autoencoder_config, autoencoder=True)
    # TODO: call run experiment

    run_experiment(autoencoder_hp_dict)


if __name__ == "__main__":
    # TODO: all the systems stuff, like experiment tags, goes into arg parser

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_exp", type=str, help="Add a unique label to show this experiment belongs to you")
    args = parser.parse_args()

    # TODO: start a modelCache instance to start auto recording the losses

    # TODO: argument related to network arch goes to


    main(sys.argv[1])
