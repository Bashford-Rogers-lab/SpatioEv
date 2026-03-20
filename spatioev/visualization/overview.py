import scanpy as sc

def plot_refinement_umaps(results):

    for name, adata in results.items():

        sc.pl.umap(
            adata,
            color="leiden",
            title=name
        )