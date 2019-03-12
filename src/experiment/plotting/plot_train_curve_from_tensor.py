import matplotlib.pyplot as plt
import csv
import os
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (8, 4)


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



if __name__ == "__main__":

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