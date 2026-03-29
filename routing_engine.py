"""
routing_engine.py — Phase 4: Road Graph & Routing Engine
==========================================================
Builds a city road network using NetworkX, with hospital nodes
sourced from hospital_data.csv. Implements shortest-path routing,
hospital finder, and greedy ambulance dispatch.

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_HOSPITALS = 10
NUM_DEPOTS = 3
NUM_INTERSECTIONS = 37  # total = 10 + 3 + 37 = 50
TARGET_EDGES = 120
GRAPH_PATH = "road_network.gpickle"
MAP_PATH = "road_network_map.png"
HOSPITAL_CSV = "hospital_data.csv"


def build_road_network(hospital_df: pd.DataFrame,
                       seed: int = SEED) -> nx.Graph:
    """
    Build a synthetic city road network with 50 nodes and ~120 edges.

    Node types:
      - hospital    (10)  — mapped to hospital_data.csv IDs
      - depot       (3)   — ambulance stations
      - intersection (37) — road intersections

    Parameters
    ----------
    hospital_df : pd.DataFrame
        Hospital metadata (must have hospital_id, latitude, longitude).
    seed : int
        Random seed.

    Returns
    -------
    nx.Graph
        Weighted graph (edge weight = travel_time_min in minutes).
    """
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    node_id = 0

    # --- Hospital nodes ---
    for i in range(min(NUM_HOSPITALS, len(hospital_df))):
        row = hospital_df.iloc[i]
        G.add_node(node_id,
                   name=row["hospital_name"],
                   node_type="hospital",
                   hospital_id=row["hospital_id"],
                   latitude=float(row["latitude"]),
                   longitude=float(row["longitude"]))
        node_id += 1

    # --- Ambulance depot nodes ---
    depot_names = ["Central Depot", "North Depot", "South Depot"]
    for i in range(NUM_DEPOTS):
        G.add_node(node_id,
                   name=depot_names[i],
                   node_type="depot",
                   latitude=round(rng.uniform(17.32, 17.48), 6),
                   longitude=round(rng.uniform(78.37, 78.53), 6))
        node_id += 1

    # --- Intersection nodes ---
    for i in range(NUM_INTERSECTIONS):
        G.add_node(node_id,
                   name=f"Intersection_{i + 1}",
                   node_type="intersection",
                   latitude=round(rng.uniform(17.30, 17.50), 6),
                   longitude=round(rng.uniform(78.35, 78.55), 6))
        node_id += 1

    # --- Edges: spanning tree first (guarantees connectivity),
    #     then random shortcuts up to TARGET_EDGES ---
    nodes_list = list(G.nodes())
    shuffled = nodes_list.copy()
    rng.shuffle(shuffled)

    for i in range(1, len(shuffled)):
        w = round(rng.uniform(1, 15), 1)
        G.add_edge(shuffled[i - 1], shuffled[i], travel_time_min=w)

    attempts = 0
    while G.number_of_edges() < TARGET_EDGES and attempts < 2000:
        u, v = rng.choice(nodes_list, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v, travel_time_min=round(rng.uniform(1, 15), 1))
        attempts += 1

    print(f"[INFO] Road network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def save_graph(G: nx.Graph, path: str = GRAPH_PATH):
    """Save graph to pickle."""
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"[✓] Graph saved → {path}")


def load_graph(path: str = GRAPH_PATH) -> nx.Graph:
    """Load graph from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def get_shortest_path(graph: nx.Graph, source_node: int,
                      target_node: int) -> dict:
    """
    Find shortest path (Dijkstra) between two nodes.

    Returns
    -------
    dict : {"path": list, "total_time_min": float, "hops": int}
    """
    try:
        path = nx.dijkstra_path(graph, source_node, target_node,
                                weight="travel_time_min")
        cost = nx.dijkstra_path_length(graph, source_node, target_node,
                                       weight="travel_time_min")
        return {"path": path, "total_time_min": round(cost, 2),
                "hops": len(path) - 1}
    except nx.NetworkXNoPath:
        return {"path": [], "total_time_min": float("inf"), "hops": 0}


def find_nearest_hospital(graph: nx.Graph, incident_node: int,
                          hospital_data_df: pd.DataFrame,
                          top_n: int = 3) -> list:
    """
    Find the top-N nearest hospitals ranked by composite score
    (travel time + bed availability + trauma centre bonus).

    Returns
    -------
    list[dict]
    """
    hospital_nodes = [
        (n, d) for n, d in graph.nodes(data=True)
        if d.get("node_type") == "hospital"
    ]

    results = []
    for nid, ndata in hospital_nodes:
        route = get_shortest_path(graph, incident_node, nid)
        hid = ndata.get("hospital_id", "")

        h_row = hospital_data_df[hospital_data_df["hospital_id"] == hid]
        if len(h_row) > 0:
            h = h_row.iloc[0]
            available_beds = int(h["total_beds"] - h["current_occupancy"])
            is_trauma = bool(h["trauma_center"])
        else:
            available_beds = 0
            is_trauma = False

        travel = route["total_time_min"]
        score = travel - max(0, available_beds) * 0.05 - (3 if is_trauma else 0)

        results.append({
            "hospital_id": hid,
            "hospital_name": ndata.get("name", "Unknown"),
            "node_id": nid,
            "travel_time_min": travel,
            "available_beds": available_beds,
            "trauma_center": is_trauma,
            "score": round(score, 2),
        })

    results.sort(key=lambda x: x["score"])
    return results[:top_n]


def assign_ambulance(incident_node: int,
                     available_ambulances: list,
                     graph: nx.Graph) -> dict:
    """
    Greedy ambulance dispatch — assign closest available unit.

    Parameters
    ----------
    incident_node : int
    available_ambulances : list[dict]
        Each: {"ambulance_id": str, "current_node": int}
    graph : nx.Graph

    Returns
    -------
    dict : {"ambulance_id", "eta_min", "route"}
    """
    best = None
    for amb in available_ambulances:
        route = get_shortest_path(graph, amb["current_node"], incident_node)
        if best is None or route["total_time_min"] < best["eta_min"]:
            best = {
                "ambulance_id": amb["ambulance_id"],
                "eta_min": route["total_time_min"],
                "route": route["path"],
            }
    return best or {"ambulance_id": "NONE", "eta_min": float("inf"), "route": []}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_network(graph: nx.Graph, save_path: str = MAP_PATH):
    """Plot colour-coded road network and save as PNG."""
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = {n: (d.get("longitude", 0), d.get("latitude", 0))
           for n, d in graph.nodes(data=True)}

    cmap = {"intersection": "#4A90D9", "hospital": "#E74C3C", "depot": "#27AE60"}
    colors = [cmap.get(graph.nodes[n].get("node_type"), "#4A90D9")
              for n in graph.nodes()]

    ew = [graph[u][v].get("travel_time_min", 5) for u, v in graph.edges()]
    max_w = max(ew) if ew else 1
    widths = [0.5 + 2.5 * (1 - w / max_w) for w in ew]

    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths,
                           alpha=0.4, edge_color="#888888")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors,
                           node_size=120, edgecolors="black", linewidths=0.5)

    labels = {n: d.get("name", "")[:15]
              for n, d in graph.nodes(data=True)
              if d.get("node_type") in ("hospital", "depot")}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax,
                            font_size=6, font_weight="bold")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90D9',
               markersize=10, label='Intersection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='Hospital'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
               markersize=10, label='Ambulance Depot'),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title("City Road Network — Emergency Response", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Road network map saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 4] Road Graph & Routing Engine")
    print("=" * 50)

    hospital_df = pd.read_csv(HOSPITAL_CSV)
    print(f"[INFO] Loaded {len(hospital_df)} hospitals")

    G = build_road_network(hospital_df)
    save_graph(G)
    visualise_network(G)

    # --- Test dispatch ---
    print("\n--- Test Dispatch Scenario ---")
    intersections = [n for n, d in G.nodes(data=True)
                     if d.get("node_type") == "intersection"]
    incident = intersections[5]

    top_h = find_nearest_hospital(G, incident, hospital_df, top_n=3)
    print(f"\nIncident at node {incident} ({G.nodes[incident].get('name')})")
    print("Top 3 hospitals:")
    for i, h in enumerate(top_h, 1):
        print(f"  {i}. {h['hospital_name']} — {h['travel_time_min']} min, "
              f"{h['available_beds']} beds, score={h['score']}")

    depots = [n for n, d in G.nodes(data=True) if d.get("node_type") == "depot"]
    ambulances = [{"ambulance_id": f"AMB{i+1:02d}", "current_node": depots[i]}
                  for i in range(len(depots))]

    dispatch = assign_ambulance(incident, ambulances, G)
    print(f"\nDispatched: {dispatch['ambulance_id']} — "
          f"ETA: {dispatch['eta_min']} min")

    print("\nRouting engine ready.")
