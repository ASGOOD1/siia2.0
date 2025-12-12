import pandas as pd
import numpy as np
import pickle
import os
from NeuralNetwork import NeuralNetwork

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm


def preprocess(df):
    df = df.replace("?", np.nan)
    for col in ["media_s1", "absente", "comportament"]:
        df[col] = pd.to_numeric(df[col])
        df[col] = df[col].fillna(df[col].median())

    df["environment"] = df["environment"].map({"urban": 1, "rural": 0})

    scalers = {}
    for col in ["media_s1", "absente", "comportament"]:
        col_min = df[col].min()
        col_max = df[col].max()
        scalers[col] = (col_min, col_max)
        denom = col_max - col_min
        if denom != 0:
            df[col] = (df[col] - col_min) / denom
        else:
            df[col] = 0.0

   
    X = df[["environment", "media_s1", "absente", "comportament"]].values
    Y = pd.to_numeric(df["trece"], errors="coerce").fillna(0).astype(float).values
    return X, Y, scalers


def train_and_save(csv_path='elevi_10000.csv', model_path='model.pkl', scalers_path='scalers.pkl', epochs=1000, lr=0.01, batch_size=32):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    X, Y, scalers = preprocess(df)

    net = NeuralNetwork()

    # Live plotting setup: loss + network graph showing activations and weights
    monitor_size = min(200, X.shape[0])
    monitor_X = X[:monitor_size]

    epochs_list = []
    losses = []

    plt.ion()  # Commented out for non-interactive backend
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[1, :])

    loss_line, = ax_loss.plot([], [], '-o')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training loss')

    # prepare network graph structure once
    G = nx.DiGraph()
    # input layer nodes
    input_n = net.l1.input_dim
    l1_n = net.l1.num_neurons
    l2_n = net.l2.num_neurons
    out_n = net.l3.num_neurons

    input_nodes = [f'in{i}' for i in range(input_n)]
    l1_nodes = [f'h1_{i}' for i in range(l1_n)]
    l2_nodes = [f'h2_{i}' for i in range(l2_n)]
    out_nodes = [f'out0']

    for n in input_nodes + l1_nodes + l2_nodes + out_nodes:
        G.add_node(n)

    # edges: inputs -> l1
    for i in range(input_n):
        for j in range(l1_n):
            G.add_edge(input_nodes[i], l1_nodes[j])
    # l1 -> l2
    for i in range(l1_n):
        for j in range(l2_n):
            G.add_edge(l1_nodes[i], l2_nodes[j])
    # l2 -> out
    for i in range(l2_n):
        G.add_edge(l2_nodes[i], out_nodes[0])

    # layout positions by layer
    pos = {}
    # x positions per layer
    layers_x = { 'in': 0.0, 'h1': 0.33, 'h2': 0.66, 'out': 1.0 }
    for idx, n in enumerate(input_nodes):
        pos[n] = (layers_x['in'], 1.0 - idx * (1.0 / max(1, input_n-1))) if input_n>1 else (layers_x['in'], 0.5)
    for idx, n in enumerate(l1_nodes):
        pos[n] = (layers_x['h1'], 1.0 - idx * (1.0 / max(1, l1_n-1))) if l1_n>1 else (layers_x['h1'], 0.5)
    for idx, n in enumerate(l2_nodes):
        pos[n] = (layers_x['h2'], 1.0 - idx * (1.0 / max(1, l2_n-1))) if l2_n>1 else (layers_x['h2'], 0.5)
    pos[out_nodes[0]] = (layers_x['out'], 0.5)

    cmap = cm.get_cmap('RdYlBu')

    def draw_network(activations, weights):
        # activations: dict mapping node -> value (0..1 or -1..1)
        # weights: list of (u,v,weight)
        ax_graph.clear()
        # node colors by activation (normalize to 0..1 for colormap)
        vals = []
        for n in G.nodes():
            v = activations.get(n, 0.0)
            # map tanh outputs (-1..1) to 0..1
            v_norm = (v + 1) / 2
            vals.append(v_norm)
        node_colors = [cmap(v) for v in vals]

        # edge widths by absolute weight
        edge_weights = [abs(w) for (_, _, w) in weights]
        if len(edge_weights) == 0:
            edge_widths = []
        else:
            maxw = max(edge_weights)
            edge_widths = [1 + 4 * (abs(w) / maxw) if maxw>0 else 1 for w in edge_weights]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax_graph, node_size=400)
        # draw edges in same order as weights
        edges = [(u, v) for (u, v, _) in weights]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, alpha=0.7, ax=ax_graph)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_graph)
        ax_graph.set_title('Network activations (mean) and weight magnitudes')
        ax_graph.axis('off')

    def on_epoch(e, loss):
        epochs_list.append(e)
        losses.append(loss)

        # compute activations on monitor set (mean per neuron)
        h1, h2, y = net.forward(monitor_X)
        h1_mean = np.mean(h1, axis=0)
        h2_mean = np.mean(h2, axis=0)
        y_mean = np.mean(y)

        # build activations dict
        activations = {}
        for i, n in enumerate(input_nodes):
            activations[n] = np.mean(monitor_X[:, i])  # input mean
        for i, n in enumerate(l1_nodes):
            activations[n] = h1_mean[i]
        for i, n in enumerate(l2_nodes):
            activations[n] = h2_mean[i]
        activations[out_nodes[0]] = y_mean

        # collect weights in same edge order: inputs->l1, l1->l2, l2->out
        weights = []
        # inputs->l1
        W1 = net.l1.W  # shape (input_dim, l1_n)
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                weights.append((input_nodes[i], l1_nodes[j], float(W1[i, j])))
        # l1->l2
        W2 = net.l2.W
        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                weights.append((l1_nodes[i], l2_nodes[j], float(W2[i, j])))
        # l2->out
        W3 = net.l3.W
        for i in range(W3.shape[0]):
            weights.append((l2_nodes[i], out_nodes[0], float(W3[i, 0])))

        # update loss plot
        loss_line.set_xdata(epochs_list)
        loss_line.set_ydata(losses)
        ax_loss.relim(); ax_loss.autoscale_view()

        # draw network
        draw_network(activations, weights)

        fig.canvas.draw()  # Commented out for non-interactive backend
        fig.canvas.flush_events()  # Commented out for non-interactive backend

    net.train(X, Y, epochs=epochs, lr=lr, batch_size=batch_size, on_epoch_end=on_epoch)

    # finalize plot
    plt.ioff()  # Commented out for non-interactive backend
    fig.tight_layout()
    plot_path = os.path.splitext(model_path)[0] + '_training_loss.png'
    fig.savefig(plot_path)

    with open(model_path, 'wb') as f:
        pickle.dump(net, f)
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)

    print(f"Saved model to {model_path} and scalers to {scalers_path}")
    print(f"Saved training plot to {plot_path}")
    return model_path, scalers_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train neural network and save model + scalers')
    parser.add_argument('--csv', default='elevi_10000.csv')
    parser.add_argument('--model', default='model.pkl')
    parser.add_argument('--scalers', default='scalers.pkl')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=32)

    args = parser.parse_args()
    train_and_save(csv_path=args.csv, model_path=args.model, scalers_path=args.scalers, epochs=args.epochs, lr=args.lr, batch_size=args.batch)
