import os
import itertools
import numpy as np
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import HdpModel
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import centrality
import adjustText
from adjustText import adjust_text
from tqdm.notebook import tqdm
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo



def load_hdp_models(output_directory, hdp_model_file):
    hdp_model = HdpModel.load(os.path.join(output_directory, hdp_model_file))
    return hdp_model

def extract_top_words(hdp_model, dictionary, num_words=10):
    top_words = []

    for topic_id in range(len(hdp_model.get_topics())):
        topic_terms = hdp_model.show_topic(topic_id, topn=num_words)
        words = [word for word, _ in topic_terms]
        top_words.extend(words)

    return set(top_words)


def build_cooccurrence_matrix(top_words, hdp_models, dictionaries):
    word_indices = {word: idx for idx, word in enumerate(top_words)}
    cooccurrence_matrix = np.zeros((len(top_words), len(top_words)))

    for hdp_model, dictionary in zip(hdp_models, dictionaries):
        for topic_id in range(len(hdp_model.get_topics())):
            topic_terms = hdp_model.show_topic(topic_id)
            words = [word for word, _ in topic_terms]
            word_combinations = list(itertools.combinations(words, 2))

            for word1, word2 in word_combinations:
                if word1 in word_indices and word2 in word_indices:
                    idx1 = word_indices[word1]
                    idx2 = word_indices[word2]
                    cooccurrence_matrix[idx1, idx2] += 1
                    cooccurrence_matrix[idx2, idx1] += 1

    return cooccurrence_matrix, word_indices

def build_weighted_graph(cooccurrence_matrix, word_indices):
    graph = nx.Graph()

    for word, idx in word_indices.items():
        graph.add_node(word)

    for word1, idx1 in word_indices.items():
        for word2, idx2 in word_indices.items():
            if idx1 != idx2 and cooccurrence_matrix[idx1, idx2] > 0:
                graph.add_edge(word1, word2, weight=cooccurrence_matrix[idx1, idx2])

    return graph

def analyze_graph_properties(graph):
    degree_centrality = centrality.degree_centrality(graph)
    betweenness_centrality = centrality.betweenness_centrality(graph)
    clustering_coefficient = nx.clustering(graph)

    return degree_centrality, betweenness_centrality, clustering_coefficient


from tqdm.notebook import tqdm


def visualize_graph(graph):
    pos = nx.spring_layout(graph, seed=42)
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    pyo.plot(fig, filename='/Users/vinay/PycharmProjects/pythonTextProcessor/Topic Modelling/network_graph5.html', auto_open=False)

def main():
    output_directory = "/Users/vinay/PycharmProjects/pythonTextProcessor/data"

    # Load trained HDP models
    biomech_hdp_model = load_hdp_models(output_directory, "biomech_hdp_model.hdp")
    biochem_hdp_model = load_hdp_models(output_directory, "biochem_hdp_model.hdp")

    # Load dictionaries
    biomech_dictionary = Dictionary.load(os.path.join(output_directory, "biomech_dictionary.dict"))
    biochem_dictionary = Dictionary.load(os.path.join(output_directory, "biochem_dictionary.dict"))

    # Extract top words from each topic
    top_words_biomech = extract_top_words(biomech_hdp_model, biomech_dictionary)
    top_words_biochem = extract_top_words(biochem_hdp_model, biochem_dictionary)
    top_words = top_words_biomech.union(top_words_biochem)

    # Build co-occurrence matrix
    cooccurrence_matrix, word_indices = build_cooccurrence_matrix(top_words, [biomech_hdp_model, biochem_hdp_model],
                                                                  [biomech_dictionary, biochem_dictionary])

    # Build graph
    graph = build_weighted_graph(cooccurrence_matrix, word_indices)

    # Define the list of selected words
    selected_words = ['kinase']

    # Find all nodes connected to the selected words
    connected_nodes = set()
    for word in selected_words:
        if word in graph:
            connected_nodes.add(word)
            connected_nodes.update(graph.neighbors(word))

    # Create a subgraph with the connected nodes
    subgraph = graph.subgraph(connected_nodes)

    # Visualize the subgraph
    visualize_graph(subgraph)

if __name__ == "__main__":
    main()