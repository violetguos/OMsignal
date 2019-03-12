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

if __name__ == "__main__":
    x, y = csv_to_list()
    plt.plot(x, y, label="Training accuracy")
    plt.xlabel("Number of epochs")
    plt.ylabel("Classification accuracy (%) \n / Cross-entropy loss")
    plt.title("ConvAE with FFT")
    plt.legend()
    plt.show()
