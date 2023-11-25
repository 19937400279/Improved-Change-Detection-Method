import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib as mpl

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.tick_params(labelsize=18)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=22)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=22)
    ax.tick_params(axis='y', pad=15)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels())


    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):


    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=24, **kw)
            texts.append(text)

    return texts

if __name__ == '__main__':
    sm_path = r'D:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'D:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'D:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    vh_path = r'D:\04 Method\01 Table\vh_2017_2021_sort.xlsx'
    ndvi_path = r'D:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'

    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    ts_df = pd.read_excel(ts_path, index_col=0, header=0)
    vv_df = pd.read_excel(vv_path, index_col=0, header=0)
    vh_df = pd.read_excel(vh_path, index_col=0, header=0)
    ndvi_df = pd.read_excel(ndvi_path, index_col=0, header=0)

    column_name = sm_df.columns

    sm = np.array(sm_df[column_name].values).reshape(1, -1)
    ts = np.array(ts_df[column_name].values).reshape(1, -1)
    vv = np.array(vv_df[column_name].values).reshape(1, -1)
    vh = np.array(vh_df[column_name].values).reshape(1, -1)
    ndvi = np.array(ndvi_df[column_name].values).reshape(1, -1)

    var = np.vstack((sm, ts, vv, vh, ndvi))
    correlation_matrix = np.corrcoef(var)
    labels = ["Soil Moisture", "Soil Temperature", "VV Backscatter", "VH Backscatter", "MODIS NDVI"]

    plt.rcParams['figure.figsize'] = (15, 12)  # 1920x1080
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()

    # im, cbar = heatmap(correlation_matrix, labels, labels, ax=ax, cmap="YlGn", cbarlabel='Pearson correlation coefficient')
    im, cbar = heatmap(correlation_matrix, labels, labels, ax=ax, cmap="YlGn", )
    texts = annotate_heatmap(im)

    fig.tight_layout()
    # plt.show()
    plt.savefig(r'D:\05 Essay\小论文\论文图\图5 热力图/' + '热力图.png', dpi=300)
