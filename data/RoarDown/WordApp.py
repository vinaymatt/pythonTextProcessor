import pickle
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.subplots as sp
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

thresholds = {
    2002: 0.5,
    2003: 0.33,
    2004: 0.33,
    2005: 0.33,
    2006: 0.34,
    2007: 0.35,
    2008: 0.36,
    2009: 0.37,
    2010: 0.4,
    2011: 0.4,
    2012: 0.4,
    2013: 0.4,
    2014: 0.4,
    2015: 0.4,
    2016: 0.4,
    2017: 0.4,
    2018: 0.4
}


def get_node_edge_positions(G):
    pos = nx.spring_layout(G, seed=42, weight='weight', iterations=100)
    node_x = []
    node_y = []
    node_text = []
    edge_positions = []

    for node, coords in pos.items():
        node_x.append(coords[0])
        node_y.append(coords[1])
        node_text.append(node)

    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_positions.append(((x0, x1), (y0, y1)))

    return node_x, node_y, node_text, edge_positions


def get_most_similar_words(word_vectors, word, threshold, restrict_vocab=1000):
    try:
        most_similar_words = word_vectors.similar_by_word(word, topn=restrict_vocab)
        selected_words = [(word, dist) for word, dist in most_similar_words if dist > threshold]
        print(len(selected_words))
        print(threshold)
        return selected_words
    except KeyError:
        return []


def get_year_data(year, thresholds, input_word='protein', graph_type=nx.Graph, restrict_vocab=10000):
    embeddings_file = f"{year}_word2vec_embeddings.pkl"
    threshold = thresholds[year]

    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    word_vectors = embeddings

    most_similar_words = get_most_similar_words(word_vectors, input_word, threshold, restrict_vocab)

    G = graph_type()

    if most_similar_words:
        G.add_node(input_word)
        for word, dist in most_similar_words:
            G.add_edge(input_word, word, weight=dist)

        # Check for connections between the most similar words themselves
        word_list = [word for word, _ in most_similar_words]
        vectors = [word_vectors[word] for word in word_list]
        normalized_vectors = np.array(vectors) / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        similarities = np.dot(normalized_vectors, normalized_vectors.T)

        for i, word1 in enumerate(word_list):
            for j, word2 in enumerate(word_list):
                if i != j and similarities[i, j] > threshold:
                    G.add_edge(word1, word2, weight=similarities[i, j])
    else:
        print(f"No similar words found for {input_word}")

    return G


input_word = 'mechanobiology'
years = list(range(2002, 2019))
graph_type = nx.Graph  # Use nx.Graph for undirected graph

# Get data for all years
year_data = []
for year in tqdm(years, desc="Getting data"):
    year_data.append(get_year_data(year, thresholds, input_word, graph_type))


def draw_graph_plotly(G, title):
    node_x, node_y, node_text, edge_positions = get_node_edge_positions(G)
    edge_x = []
    edge_y = []

    for edge_position in edge_positions:
        edge_x += [edge_position[0][0], edge_position[0][1], None]
        edge_y += [edge_position[1][0], edge_position[1][1], None]

    node_degrees = [G.degree(n) for n in G.nodes]
    if node_degrees:
        max_degree = max(node_degrees)
    else:
        max_degree = 1
    node_colors = node_degrees
    node_sizes = [20 + 50 * (degree / max_degree) for degree in node_degrees]

    # Create edge trace
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    # Create node trace
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_text,
                            marker=dict(showscale=True, colorscale='Viridis', reversescale=True, color=node_colors,
                                        size=node_sizes,
                                        colorbar=dict(thickness=15, title='Node Connections', xanchor='left',
                                                      titleside='right'), line_width=2))

    # Create a figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update the figure layout
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, b=0, t=35, pad=4),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


# Call the new function to generate the visualization with a slider
# draw_graph_plotly_with_slider(year_data, years, f"word_network_{input_word}_slider1.html", f"Word Network for {input_word}")

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Create a function to generate the sorted node list for a given graph
def get_sorted_node_list(G):
    return sorted(G.nodes, key=lambda x: G.degree(x), reverse=True)


# Prepare the initial data for the first year
initial_year = years[0]
initial_G = year_data[0]
initial_sorted_nodes = get_sorted_node_list(initial_G)

# Create the layout for the Dash app
app.layout = html.Div([
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in years],
        value=initial_year
    ),
    html.Div(id='sorted-node-list'),
    dcc.Graph(id='network-graph'),
])


# Update the sorted node list and network graph when the selected year changes
@app.callback(
    [Output('sorted-node-list', 'children'),
     Output('network-graph', 'figure')],
    [Input('year-dropdown', 'value')]
)
def update_data(selected_year):
    year_index = years.index(selected_year)
    G = year_data[year_index]
    sorted_nodes = get_sorted_node_list(G)

    # Generate the network graph using the draw_graph_plotly function
    fig = draw_graph_plotly(G, f"Word Network for {input_word} ({selected_year})")

    # Create a sorted node list with click-to-highlight functionality
    sorted_node_list = dbc.ListGroup([
        dbc.ListGroupItem(node, id={'type': 'node-item', 'index': index}, action=True)
        for index, node in enumerate(sorted_nodes)
    ])

    return sorted_node_list, fig


# Highlight the selected node in the network graph
@app.callback(
    Output('network-graph', 'figure'),
    [Input({'type': 'node-item', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('year-dropdown', 'value'),
     State('network-graph', 'figure')]
)
def highlight_node(n_clicks_list, selected_year, current_fig):
    if not any(n_clicks_list):
        return current_fig

    clicked_node_index = n_clicks_list.index(max(n_clicks_list))
    year_index = years.index(selected_year)
    G = year_data[year_index]
    sorted_nodes = get_sorted_node_list(G)
    highlighted_node = sorted_nodes[clicked_node_index]

    new_fig = draw_graph_plotly(G, f"Word Network for {input_word} ({selected_year})")

    for i, d in enumerate(new_fig.data[1]['marker']['color']):
        if new_fig.data[1]['hovertext'][i] == highlighted_node:
            new_fig.data[1]['marker']['color'][i] = '#FF0000'
            new_fig.data[1]['marker']['size'][i] = new_fig.data[1]['marker']['size'][i] * 1.5

    return new_fig


# Run the Dash app
app.run_server()