import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
import networkx as nx
import plotly.graph_objects as go
from tqdm import tqdm


def create_adjacency_matrix(similarity_matrix, threshold, min_closest_words=30):
    adjacency_matrix = (similarity_matrix > threshold).astype(int)

    # Get the indices of the top min_closest_words closest words for each word
    closest_word_indices = np.argpartition(similarity_matrix, -min_closest_words - 1, axis=1)[:,
                           -min_closest_words - 1:-1]

    # Set the values in the adjacency matrix to 1 for these indices
    for i, row in enumerate(closest_word_indices):
        adjacency_matrix[i, row] = 1

    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


def get_word_subgraph(G, word, word_index_mapping):
    word_index = word_index_mapping[word]
    neighbors = [n for n in G.neighbors(word_index)]
    word_subgraph = G.subgraph([word_index] + neighbors)
    return word_subgraph


def get_word_appearance_year(years, input_word):
    for i, year in enumerate(years):
        with open(f"{year}_tokenized_abstracts_bigram.pkl", "rb") as f:
            tokenized_abstracts_bigram = pickle.load(f)
        if input_word in [word for abstract in tokenized_abstracts_bigram for word in abstract]:
            return i
    return None


def get_year_data(year, threshold, input_word, min_closest_words):
    embeddings_file = f"{year}_word2vec_embeddings.pkl"
    tokenized_abstracts_bigram_file = f"{year}_tokenized_abstracts_bigram.pkl"

    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    with open(tokenized_abstracts_bigram_file, "rb") as f:
        tokenized_abstracts_bigram = pickle.load(f)

    word_vectors = embeddings

    if input_word in word_vectors.key_to_index:
        # Query the most similar words
        most_similar_words = word_vectors.most_similar(input_word, topn=100)
        relevant_words = [w for w, sim in most_similar_words if sim > threshold]
        if len(relevant_words) < min_closest_words:
            relevant_words = [w for w, sim in most_similar_words][:min_closest_words]
    else:
        # If the input word is not in the dictionary, create an empty list
        relevant_words = []

    relevant_words.append(input_word)

    relevant_embeddings = {}
    for word in relevant_words:
        try:
            relevant_embeddings[word] = word_vectors[word]
        except KeyError:
            pass

    # Check if the relevant_embeddings dictionary is empty
    if not relevant_embeddings:
        return [], [], [], [], []

    embedding_matrix = np.array([vec for vec in relevant_embeddings.values()])
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Update the threshold to the 50th percentile of the similarity matrix
    threshold = np.percentile(similarity_matrix, 50)

    adjacency_matrix = create_adjacency_matrix(similarity_matrix, threshold, min_closest_words)

    G = nx.from_numpy_array(adjacency_matrix)

    word_index_mapping = {i: word for i, word in enumerate(relevant_words)}
    index_word_mapping = {word: i for i, word in word_index_mapping.items()}
    nx.set_node_attributes(G, word_index_mapping, "label")

    word_subgraph = get_word_subgraph(G, input_word, index_word_mapping)

    pos = nx.spring_layout(word_subgraph, seed=42)

    node_x = [pos[i][0] for i in word_subgraph.nodes]
    node_y = [pos[i][1] for i in word_subgraph.nodes]
    node_text = [word_subgraph.nodes[i]["label"] for i in word_subgraph.nodes]

    edge_x = []
    edge_y = []
    for edge in word_subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    return node_x, node_y, node_text, edge_x, edge_y


threshold = 0.5
input_word = 'mechanobiology'
years = list(range(1990, 2015))

# Get the index of the first year the input word appears
first_year_idx = get_word_appearance_year(years, input_word)

# Get data for all years
year_data = []
for i, year in enumerate(tqdm(years, desc="Getting data")):
    if first_year_idx is not None and i >= first_year_idx:
        min_closest_words = min(10 + 10 * (i - first_year_idx), 30)
    else:
        min_closest_words = 30
    year_data.append(get_year_data(year, threshold, input_word, min_closest_words))

# Create the initial frame (year 1990)
node_x, node_y, node_text, edge_x, edge_y = year_data[0]

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    hoverinfo="text",
    marker=dict(size=10, color="skyblue"),
    name="Nodes"
)

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(color="gray", width=1),
    hoverinfo="none",
    name="Edges"
)

frames = []

# Add frames for each year
for year, data in zip(years, year_data):
    node_x, node_y, node_text, edge_x, edge_y = data

    node_trace_year = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=10, color="skyblue"),
        name="Nodes"
    )

    edge_trace_year = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="gray", width=1),
        hoverinfo="none",
        name="Edges"
    )

    frame = go.Frame(
        data=[edge_trace_year, node_trace_year],
        name=str(year),
        layout=go.Layout(title=f"Word Network around '{input_word}' for {year}")
    )
    frames.append(frame)

# Create the animated Plotly figure
animation_settings = dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)
sliders = [dict(steps=[dict(method="animate", args=[[str(year)]], label=str(year)) for year in years], active=0)]

fig = go.Figure(data=[edge_trace, node_trace], frames=frames, layout=go.Layout(updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None, animation_settings])])], sliders=sliders))

# Save the animation as an HTML file
fig.write_html(f"word_network_animati1on_{input_word}.html")
