import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import community as community_louvain


def _fiber_node_id(fiber_id):
    """
    Canonical graph node id for a fiber.
    """
    return ("fiber", fiber_id)


def _cell_node_id(cell_id):
    """
    Canonical graph node id for a cell.
    """
    return ("cell", cell_id)


def build_ecm_bipartite_graph_per_image(
    adata,
    fiber_df,
    links_df,
    image_key="imageid",
    link_image_key="imageid",
    distance_scale=50,
    include_cell_attrs=True,
    include_fiber_attrs=True,
):
    """
    Build bipartite graphs per image.

    Each graph contains two node types:
    - ``("cell", cell_id)`` for cells
    - ``("fiber", fiber_id)`` for fibers

    Edges represent precomputed spatial links from ``links_df``.
    The edge weight decays with distance as ``exp(-distance / distance_scale)``.

    Parameters
    ----------
    image_key : str
        Column shared by ``adata.obs`` and ``fiber_df`` identifying each image.
    link_image_key : str
        Column in ``links_df`` identifying each image.
    distance_scale : float
        Distance-decay scale used to convert link distance into edge weight.
    include_cell_attrs, include_fiber_attrs : bool
        If ``True``, copy row attributes from the source table onto graph nodes.

    Returns
    -------
    dict
        {image_id: graph}
    """

    graphs = {}

    images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[image_key].unique()
    )

    for img in images:

        # -----------------------------
        # Subset data
        # -----------------------------
        cells = adata.obs[adata.obs[image_key] == img]
        fibers = fiber_df[fiber_df[image_key] == img]
        links = links_df[links_df[link_image_key] == img]

        G = nx.Graph()

        # -----------------------------
        # Add cell nodes
        # -----------------------------
        for cell_id in cells.index:
            attrs = {
                "node_type": "cell",
                image_key: img,
            }
            if include_cell_attrs:
                attrs.update(cells.loc[cell_id].to_dict())

            G.add_node(
                _cell_node_id(cell_id),
                **attrs,
            )

        # -----------------------------
        # Add fiber nodes
        # -----------------------------
        for fiber_id in fibers.index:
            attrs = {
                "node_type": "fiber",
                image_key: img,
            }
            if include_fiber_attrs:
                attrs.update(fibers.loc[fiber_id].to_dict())

            G.add_node(
                _fiber_node_id(fiber_id),
                **attrs,
            )

        # -----------------------------
        # Add edges (SAFE)
        # -----------------------------
        for _, row in links.iterrows():

            if row["cell_id"] not in cells.index:
                continue
            if row["fiber_id"] not in fibers.index:
                continue

            G.add_edge(
                _cell_node_id(row["cell_id"]),
                _fiber_node_id(row["fiber_id"]),
                distance=row["distance"],
                weight=np.exp(-row["distance"] / distance_scale),
                **{image_key: img},
            )

        graphs[img] = G

    return graphs

def project_fiber_graph_per_image(graphs):
    """
    Project each bipartite graph onto a fiber-only graph.

    Two fibers become connected when they share neighboring cells
    in the bipartite graph.
    """

    projected = {}

    for img, G in graphs.items():

        fiber_nodes = [
            n for n, d in G.nodes(data=True)
            if d["node_type"] == "fiber"
        ]

        if len(fiber_nodes) < 2:
            continue

        projected[img] = bipartite.weighted_projected_graph(G, fiber_nodes)

    return projected

def detect_ecm_niches_per_image(fiber_graphs, weight="weight", resolution=1.0):
    """
    Detect ECM niches separately per image.

    Runs Louvain community detection on each projected fiber graph.
    """

    niche_maps = {}

    for img, G in fiber_graphs.items():

        if len(G.nodes) < 2 or G.number_of_edges() == 0:
            continue

        partition = community_louvain.best_partition(
            G,
            weight=weight,
            resolution=resolution,
        )

        niche_maps[img] = partition

    return niche_maps


def assign_niches_to_fibers(fiber_df, niche_maps, image_key="imageid"):
    """
    Map per-image fiber-graph community labels back onto ``fiber_df``.
    """

    fiber_df = fiber_df.copy()

    niche_labels = {}

    for img, partition in niche_maps.items():
        for node, niche in partition.items():
            if not (isinstance(node, tuple) and len(node) == 2):
                continue

            node_type, fiber_id = node

            if node_type != "fiber":
                continue

            niche_labels[(img, fiber_id)] = niche

    fiber_df["niche"] = [
        niche_labels.get((img, fiber_id), np.nan)
        for fiber_id, img in zip(fiber_df.index, fiber_df[image_key])
    ]

    return fiber_df



def compute_invasion_score(
    fiber_df,
    tumor_density_col="tumor_density",
    alignment_col="alignment_score",
    image_key="imageid",
    niche_col="niche",
):
    """
    Summarize each ECM niche by tumor density, alignment, and a simple invasion score.

    Parameters
    ----------
    tumor_density_col : str
        Column in ``fiber_df`` containing tumor-density values per fiber.
    alignment_col : str
        Column in ``fiber_df`` containing fiber alignment values.
    image_key : str
        Column in ``fiber_df`` identifying each image.
    niche_col : str
        Column in ``fiber_df`` containing niche labels assigned to fibers.
    """

    df = fiber_df.copy()

    summary = (
        df.groupby([image_key, niche_col])
        [[tumor_density_col, alignment_col]]
        .mean()
        .reset_index()
    )

    summary["invasion_score"] = (
        summary[tumor_density_col] *
        summary[alignment_col]
    )

    return summary.sort_values("invasion_score", ascending=False)
