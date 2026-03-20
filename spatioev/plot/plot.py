# spatioev/plot/plot.py

import matplotlib.pyplot as plt
import seaborn as sns


def plot_area_distribution(adata, min_area=None, max_area=None):
    plt.figure(figsize=(2, 1.5))
    sns.histplot(adata.obs["area_um2"], bins=50)

    if min_area is not None:
        plt.axvline(min_area, color="red", linewidth=1)
    if max_area is not None:
        plt.axvline(max_area, color="green", linewidth=1)

    plt.title("Cell area distribution")
    plt.xlabel("Area (µm²)")
    plt.ylabel("Cell count")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_nc_ratio_distribution(adata, max_ratio=None):
    plt.figure(figsize=(2, 1.5))
    sns.histplot(adata.obs["nc_ratio"], bins=50)

    if max_ratio:
        plt.axvline(max_ratio, color="red", linewidth=1)

    plt.title("Nuclear-to-cell ratio distribution")
    plt.xlabel("NC ratio")
    plt.ylabel("Cell count")
    sns.despine()
    plt.tight_layout()
    plt.show()