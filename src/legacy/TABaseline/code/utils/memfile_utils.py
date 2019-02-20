import numpy as np

"""plots tsne beautifully using fashion_scatter"""

def read_memfile(filename, shape, dtype='float32'):
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    data = np.zeros(shape=shape, dtype=dtype)
    data[:] = fp[:]
    del fp
    return data


def write_memfile(data, filename):
    # write a numpy array 'data' into a binary  data file specified by
    # 'filename'
    shape = data.shape
    dtype = data.dtype
    fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    fp[:] = data[:]
    del fp


def fashion_scatter(x, colors):

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    %matplotlib inline
    import seaborn as sns

    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.savefig('test.png')
    plt.close()

    # return f, ax, sc, txts


if __name__ == '__main__':

    from sklearn.manifold import TSNE
    import time

    data = read_memfile('../MILA_TrainLabeledData.dat', shape=(160, 3754))
    dataSeries = data[:, :3750]

    time_start = time.time()

    fashion_tsne = TSNE(perplexity=4).fit_transform(dataSeries)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() -
                                                        time_start))

    fashion_scatter(fashion_tsne, data[:, -1])

    print('ok')
