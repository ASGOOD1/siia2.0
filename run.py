import os
import pickle
import numpy as np
from train_model import train_and_save
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib import colormaps



def load_model(model_path='model.pkl', scalers_path='scalers.pkl', csv_path='elevi_10000.csv'):
    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        print('Model or scalers not found. Training model now (this may take a while)...')
        train_and_save(csv_path=csv_path, model_path=model_path, scalers_path=scalers_path)

    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    return net, scalers


def normalize_input(values, scalers):
    # values: [environment(0/1), media, absente, comportament]
    arr = np.array(values, dtype=float).copy()
    
    # assemble full raw vector
    raw = np.array([arr[0], arr[1], arr[2], arr[3]], dtype=float)

    full = np.empty(4, dtype=float)
    full[0] = raw[0]
    cols = ['media_s1', 'absente', 'comportament']
    for i, col in enumerate(cols):
        minv, maxv = scalers[col]
        denom = maxv - minv
        val = raw[i+1]
        if denom != 0:
            full[i+1] = (val - minv) / denom
        else:
            full[i+1] = 0.0
    return full


def run_interactive():
    
    
    net, scalers = load_model()

    mediu = input("Introduceti mediul de provenienta (urban/rural): ").strip().lower()
    mediu_val = 1 if mediu == 'urban' else 0

    def safe_float_input(prompt, default=np.nan):
        val = input(prompt)
        if val.strip() == '':
            return default
        try:
            return float(val)
        except ValueError:
            return default

    medie = safe_float_input("Introduceti media semestriala (0-10): ")
    absente = safe_float_input("Introduceti numarul de absente: ")
    comportament = safe_float_input("Introduceti scorul de comportament (0-1): ")

    # If user left some blank, fall back to scaler midpoints
    if np.isnan(medie):
        medie = (scalers['media_s1'][0] + scalers['media_s1'][1]) / 2
    if np.isnan(absente):
        absente = (scalers['absente'][0] + scalers['absente'][1]) / 2
    if np.isnan(comportament):
        comportament = (scalers['comportament'][0] + scalers['comportament'][1]) / 2

    raw_input = np.array([mediu_val, medie, absente, comportament], dtype=float)
    
    elev_norm = normalize_input(raw_input, scalers)
    prob = net.predict(elev_norm)

    print('\nElev nou (raw):', raw_input)
    print('Probabilitate trecere: ~', round(prob, 3) * 100, "%")
    print('Rezultat:', 'Trece' if prob > 0.5 else 'Nu trece')
    

    # Draw network graph for this single input (activations per node)
    h1, h2, y = net.forward(elev_norm.reshape(1, -1))
    h1 = h1.flatten()
    h2 = h2.flatten()
    y = float(y.flatten()[0])

    # build graph
    G = nx.DiGraph()
    input_n = net.l1.input_dim
    l1_n = net.l1.num_neurons
    l2_n = net.l2.num_neurons
    input_nodes = [f'in{i}' for i in range(input_n)]
    l1_nodes = [f'h1_{i}' for i in range(l1_n)]
    l2_nodes = [f'h2_{i}' for i in range(l2_n)]
    out_node = 'out0'
    for n in input_nodes + l1_nodes + l2_nodes + [out_node]:
        G.add_node(n)

    for i in range(input_n):
        for j in range(l1_n):
            G.add_edge(input_nodes[i], l1_nodes[j])
    for i in range(l1_n):
        for j in range(l2_n):
            G.add_edge(l1_nodes[i], l2_nodes[j])
    for i in range(l2_n):
        G.add_edge(l2_nodes[i], out_node)

    pos = {}
    layers_x = { 'in': 0.0, 'h1': 0.33, 'h2': 0.66, 'out': 1.0 }
    for idx, n in enumerate(input_nodes):
        pos[n] = (layers_x['in'], 1.0 - idx * (1.0 / max(1, input_n-1))) if input_n>1 else (layers_x['in'], 0.5)
    for idx, n in enumerate(l1_nodes):
        pos[n] = (layers_x['h1'], 1.0 - idx * (1.0 / max(1, l1_n-1))) if l1_n>1 else (layers_x['h1'], 0.5)
    for idx, n in enumerate(l2_nodes):
        pos[n] = (layers_x['h2'], 1.0 - idx * (1.0 / max(1, l2_n-1))) if l2_n>1 else (layers_x['h2'], 0.5)
    pos[out_node] = (layers_x['out'], 0.5)

    activations = {}
    for i, n in enumerate(input_nodes):
        activations[n] = elev_norm[i]  # Use normalized 4-element vector
    for i, n in enumerate(l1_nodes):
        activations[n] = float(h1[i])
    for i, n in enumerate(l2_nodes):
        activations[n] = float(h2[i])
    activations[out_node] = y

    # weights list
    weights = []
    W1 = net.l1.W
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            weights.append((input_nodes[i], l1_nodes[j], float(W1[i, j])))
    W2 = net.l2.W
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            weights.append((l1_nodes[i], l2_nodes[j], float(W2[i, j])))
    W3 = net.l3.W
    for i in range(W3.shape[0]):
        weights.append((l2_nodes[i], out_node, float(W3[i, 0])))

    cmap = plt.colormaps.get_cmap('RdYlBu')
    vals = []
    for n in G.nodes():
        v = activations.get(n, 0.0)
        v_norm = (v + 1) / 2
        vals.append(v_norm)
    node_colors = [cmap(v) for v in vals]
    edge_weights = [abs(w) for (_, _, w) in weights]
    maxw = max(edge_weights) if len(edge_weights)>0 else 1.0
    edge_widths = [1 + 4 * (abs(w) / maxw) if maxw>0 else 1 for w in edge_weights]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax2, node_size=400)
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for (u,v,_) in weights], width=edge_widths, alpha=0.7, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
    ax2.set_title('Network activations (single input)')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_network.png', dpi=100, bbox_inches='tight')
    print('Network graph saved to prediction_network.png')


if __name__ == '__main__':
    run_interactive()
