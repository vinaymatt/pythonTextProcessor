import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
import pickle
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
import leidenalg
import igraph as ig

# Load the embeddings from the pickle file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/word2vecbiochem_embeddings.pkl', 'rb') as f:
    biochem_embeddings = pickle.load(f)

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/word2vecbiomech_embeddings.pkl', 'rb') as f:
    biomech_embeddings = pickle.load(f)

# Calculate pairwise cosine similarity between abstracts from different fields
similarity_matrix = cosine_similarity(biochem_embeddings, biomech_embeddings)

# Set a similarity threshold
similarity_threshold = 0.8

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(range(len(biochem_embeddings) + len(biomech_embeddings)))

# Add edges to the graph based on the similarity threshold
for i, row in tqdm(enumerate(similarity_matrix), total=len(similarity_matrix), desc="Adding edges"):
    for j, similarity in enumerate(row):
        if similarity > similarity_threshold:
            G.add_edge(i, j + len(biochem_embeddings), weight=similarity)

# Convert the NetworkX graph to an iGraph object
ig_graph = ig.Graph.from_networkx(G)

def leiden_with_progress(ig_graph, partition_type, progress_bar=None, n_iterations=10):
    if progress_bar is None:
        return leidenalg.find_partition(ig_graph, partition_type)

    partition = None
    for _ in range(n_iterations):
        partition = leidenalg.find_partition(ig_graph, partition_type)
        progress_bar.update(1)
        time.sleep(0.1)

    progress_bar.close()
    return partition

# Create a tqdm progress bar
progress_bar = tqdm(total=10, desc="Leiden community detection")

# Apply the Leiden community detection algorithm with progress bar
partition = leiden_with_progress(ig_graph, leidenalg.ModularityVertexPartition, progress_bar=progress_bar)

# Convert the partition object to a dictionary format compatible with NetworkX
node_partition = {}
for cluster, nodes in enumerate(tqdm(partition, desc="Converting partition")):
    for node in nodes:
        node_partition[node] = cluster

# Create a custom color map based on the field and community partition
def create_custom_color_map(G, partition, biochem_nodes, biomech_nodes):
    colors = []
    n_partitions = max(partition.values()) + 1
    color_step = 1 / n_partitions

    for node in G.nodes:
        partition_color = partition[node] * color_step
        if node in biochem_nodes:
            colors.append(plt.cm.Blues(partition_color + color_step))
        else:
            colors.append(plt.cm.Reds(partition_color + color_step))
    return colors

# Create a set of biochemistry nodes
biochem_nodes = set(range(len(biochem_embeddings)))

# Create a set of biomechanics nodes
biomech_nodes = set(range(len(biochem_embeddings), len(biochem_embeddings) + len(biomech_embeddings)))


# Create a custom color map based on the field and community partition
custom_color_map = create_custom_color_map(G, node_partition, biochem_nodes, biomech_nodes)

# Draw the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=custom_color_map, with_labels=False, node_size=30, edge_color="gray")
plt.axis("off")
plt.show()
plt.savefig('/Users/vinay/PycharmProjects/pythonTextProcessor/data/networkx.png')

# Draw the network
plt.figure(figsize=(12, 12))
pos1 = nx.kamada_kawai_layout(G, seed=42)
nx.draw(G, pos1, node_color=custom_color_map, with_labels=False, node_size=30, edge_color="gray")
plt.axis("off")
plt.show()
plt.savefig('/Users/vinay/PycharmProjects/pythonTextProcessor/data/kamada.png')
