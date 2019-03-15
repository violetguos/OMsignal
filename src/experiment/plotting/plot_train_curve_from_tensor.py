import matplotlib.pyplot as plt
import csv
import os
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (7, 5)


def csv_to_list(csv_fp="log/fft_tb/run_fft_tb-tag-Train_acc.csv"):

    x = []
    y = []

    fp = os.path.join("../../../", csv_fp)
    with open(fp, "r") as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        next(plots, None)  # skip the headers

        for row in plots:
            x.append(int(row[1]))
            y.append(float(row[2]))
    return x, y

def vae_fft_model():
    train_acc_csv_fp = "log/fft_tb/run_fft_tb-tag-Train_acc.csv"
    train_ce_csv_fp = "log/fft_tb/run_fft_tb-tag-Train_cross_entropy.csv"

    valid_acc_csv_fp = "log/fft_tb/run_fft_tb-tag-Valid_acc.csv"
    valid_ce_csv_fp = "log/fft_tb/run_fft_tb-tag-Valid_cross_entropy.csv"

    mse_loss_fp = "log/fft_tb/run_fft_tb-tag-train_mse_loss.csv"

    plot_fp = {
        'Train accuracy': train_acc_csv_fp,
        'Train loss': train_ce_csv_fp,
        'Valid accuracy': valid_acc_csv_fp,
        'Valid loss': valid_ce_csv_fp,
        'Reconstruction': mse_loss_fp
    }
    for key, val in plot_fp.items():
        x, y = csv_to_list(val)
        plt.plot(x, y, label=key)

    plt.xlabel("Number of epochs")
    plt.ylabel("Classification accuracy (%) \n / Cross-entropy loss")
    plt.title("ConvAE with FFT")
    plt.legend()
    fig = 'train_curve'
    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()


def unsupervised_pretrain():

    train_score_csv_fp = "log/final_2/run_final_2-tag-Training_OverallScore.csv"

    valid_acc_csv_fp = "log/final_2/run_final_2-tag-Valid_OverallScore.csv"


    plot_fp = {
        'Train overall score': train_score_csv_fp,
        'Valid overall score': valid_acc_csv_fp,
    }
    for key, val in plot_fp.items():
        x, y = csv_to_list(val)
        plt.plot(x, y, label=key)

    plt.xlabel("Number of epochs")
    plt.ylabel("Score")
    plt.title("Unsupervised pretraining")
    plt.legend()
    fig = 'Unsupervised_train_curve'
    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()

def final_model():

    train_score_csv_fp = "log/final/run_final-tag-Training_OverallScore.csv"
    train_loss_csv_fp = "log/final/run_final-tag-Training_Loss.csv"

    valid_score_csv_fp = "log/final/run_final-tag-Valid_OverallScore.csv"
    valid_loss_csv_fp = "log/final/run_final-tag-Valid_Loss.csv"

    mse_loss_fp = "log/final/run_final-tag-Training_ReconstructLoss.csv"

    plot_fp = {
        'Train overall score': train_score_csv_fp,
        'Train loss': train_loss_csv_fp,
        'Valid overall score': valid_score_csv_fp,
        'Valid loss': valid_loss_csv_fp,
        'Reconstruction': mse_loss_fp
    }
    for key, val in plot_fp.items():
        x, y = csv_to_list(val)
        plt.plot(x, y, label=key)

    plt.xlabel("Number of epochs")
    plt.ylabel("overall score / weighted loss")
    plt.title("Autoencoder semisupervised")
    plt.legend()
    fig = 'train_curve_final'
    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()


    plot_fp_2 = {
        'Train loss': train_loss_csv_fp,
        'Valid loss': valid_loss_csv_fp,
    }
    for key, val in plot_fp_2.items():
        x, y = csv_to_list(val)
        plt.plot(x, y, label=key)

    plt.xlabel("Number of epochs")
    plt.ylabel("overall score / weighted loss")
    plt.title("Autoencoder semisupervised")
    plt.legend()
    fig = 'train_curve_final_2'
    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()



    plot_fp_3 = {
        'Train overall score': train_score_csv_fp,
        'Valid overall score': valid_score_csv_fp,
        'Reconstruction': mse_loss_fp

    }
    for key, val in plot_fp_3.items():
        x, y = csv_to_list(val)
        plt.plot(x, y, label=key)

    plt.xlabel("Number of epochs")
    plt.ylabel("overall score / weighted loss")
    plt.title("Autoencoder semisupervised")
    plt.legend()
    fig = 'train_curve_final_2'
    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()
if __name__ == "__main__":
    unsupervised_pretrain()