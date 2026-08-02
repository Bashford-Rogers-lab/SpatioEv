"""ECM bipartite graph, niche detection, and invasion scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad
import networkx as nx
from networkx.algorithms import bipartite

# Section 3: ECM bipartite graph  (from archive/spatial/spatial_ecm_graph.py)
# ============================================================


try:
    import community as community_louvain
except ImportError:
    community_louvain = None


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
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    image_key: str="imageid",
    link_image_key: str="imageid",
    distance_scale: float=50,
    include_cell_attrs: bool=True,
    include_fiber_attrs: bool=True,
) -> pd.DataFrame:
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

def project_fiber_graph_per_image(graphs: dict) -> dict:
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


def _best_louvain_partition(G, weight="weight", resolution=1.0, random_state=None):
    """
    Run Louvain community detection with either python-louvain or NetworkX.
    """
    if community_louvain is not None:
        return community_louvain.best_partition(
            G,
            weight=weight,
            resolution=resolution,
            random_state=random_state,
        )

    if not hasattr(nx.algorithms.community, "louvain_communities"):
        raise ImportError(
            "ECM niche detection requires either the 'python-louvain' package "
            "or a NetworkX version with louvain_communities."
        )

    communities = nx.algorithms.community.louvain_communities(
        G,
        weight=weight,
        resolution=resolution,
        seed=random_state,
    )

    partition = {}
    for niche, nodes in enumerate(communities):
        for node in nodes:
            partition[node] = niche

    return partition


def detect_ecm_niches_per_image(
    fiber_graphs: dict,
    weight: float="weight",
    resolution: float=1.0,
    random_state: int=None,
) -> pd.DataFrame:
    """
    Detect ECM niches separately per image.

    Runs Louvain community detection on each projected fiber graph.

    Uses the optional ``python-louvain`` package when installed. If it is not
    available, falls back to NetworkX's built-in Louvain implementation.
    """

    niche_maps = {}

    for img, G in fiber_graphs.items():

        if len(G.nodes) < 2 or G.number_of_edges() == 0:
            continue

        partition = _best_louvain_partition(
            G,
            weight=weight,
            resolution=resolution,
            random_state=random_state,
        )

        niche_maps[img] = partition

    return niche_maps


def assign_niches_to_fibers(fiber_df: pd.DataFrame, niche_maps: dict, image_key: str="imageid") -> pd.DataFrame:
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
    fiber_df: pd.DataFrame,
    tumor_density_col: str="tumor_density",
    alignment_col: str="alignment_score",
    image_key: str="imageid",
    niche_col: str="niche",
) -> pd.DataFrame:
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



# ============================================================
