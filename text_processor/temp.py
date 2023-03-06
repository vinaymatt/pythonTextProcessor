import pickle
import numpy as np
from scipy.spatial.distance import cosine
import re
import networkx as nx
import plotly.graph_objects as go

# Load embeddings from the pkl file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/wordembeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Read special phrases
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cleanedproteinforword2vecfinal.txt', 'r') as f:
    special_phrases_raw = f.read().splitlines()
special_phrases = [tuple(phrase.split()) for phrase in special_phrases_raw]

# Find the top 10 closest words to "kinase" in the special phrases list
target_word = "actin"
target_embedding = embeddings[target_word]
word_distances = {}
for phrase in special_phrases:
    phrase_words = [re.sub('[^0-9a-zA-Zα-ωΑ-Ω]+', '_', word) for word in phrase]
    embeddings_list = [embeddings[word] for word in phrase_words if word in embeddings]
    if embeddings_list:
        phrase_embedding = np.mean(embeddings_list, axis=0)
    else:
        phrase_embedding = None

    if phrase_embedding is not None:
        if np.count_nonzero(target_embedding) == 0 or np.count_nonzero(phrase_embedding) == 0:
            distance = float('nan')
        else:
            distance = cosine(target_embedding, phrase_embedding)
        word_distances[' '.join(phrase)] = distance
    else:
        print(f"Warning: Word(s) '{phrase_words}' not found in embeddings dictionary")

top_10_words = sorted(word_distances, key=word_distances.get)[:10]

# Find the top 10 closest words for each word in the top_10_words from the entire word embedding
extended_word_distances = {}
for word in top_10_words:
    try:
        target_embedding = embeddings[word]  # Add try-except block here
    except KeyError:
        print(f"Warning: Phrase '{word}' not found in embeddings dictionary")
        continue

    word_distances = {}
    for key in embeddings.key_to_index:
        if key.isalnum():
            distance = cosine(target_embedding, embeddings[key])
            word_distances[key] = distance
    top_10_extended_words = sorted(word_distances, key=word_distances.get)[:10]
    extended_word_distances[word] = top_10_extended_words

# Create a network graph
G = nx.Graph()

# Add nodes and edges
G.add_node(target_word, color='red')
for word in top_10_words:
    G.add_node(word, color='blue')
    G.add_edge(target_word, word)

    for extended_word in extended_word_distances.get(word, []):
        G.add_node(extended_word, color='green')
        G.add_edge(word, extended_word)

# Get node positions using a layout algorithm
pos = nx.spring_layout(G)

# Create lists of node positions and colors
Xn, Yn, node_colors = [], [], []
for node, coords in pos.items():
    Xn.append(coords[0])
    Yn.append(coords[1])
    node_colors.append(G.nodes[node]['color'])

# Create lists of edge positions
Xe, Ye = [], []
for edge in G.edges():
    Xe += [pos[edge[0]][0], pos[edge[1]][0], None]
    Ye += [pos[edge[0]][1], pos[edge[1]][1], None]

# Create a Plotly Scatter trace for edges
edge_trace = go.Scatter(
    x=Xe, y=Ye,
    mode='lines',
    line=dict(color='gray', width=1),
    hoverinfo='none'
)

# Create a Plotly Scatter trace for nodes
node_trace = go.Scatter(
    x=Xn, y=Yn,
    mode='markers+text',
    marker=dict(symbol='circle', size=10, color=node_colors),
    text=list(pos.keys()),
    textposition='top center',
    textfont=dict(size=8),
    hoverinfo='text'
)

# Create a Plotly Figure and add the traces
fig = go.Figure()
fig.add_trace(edge_trace)
fig.add_trace(node_trace)

# Customize the figure appearance
fig.update_layout(
    title=f"Network Graph for '{target_word}' and Related Words",
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white'
)

fig.write_html("/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/network_graph2.html")