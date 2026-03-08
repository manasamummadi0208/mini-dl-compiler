import networkx as nx
import matplotlib.pyplot as plt
from ir.ir_graph import IRGraph


def plot_graph(graph: IRGraph, title: str = "IR Graph") -> None:
    dg = nx.DiGraph()

    for node in graph.live_nodes():
        label = f"{node.name}\n{node.op_type}"
        dg.add_node(node.name, label=label)
        for inp in node.inputs:
            if inp in graph.nodes and not graph.nodes[inp].deleted:
                dg.add_edge(inp, node.name)

    pos = nx.spring_layout(dg, seed=42)
    labels = nx.get_node_attributes(dg, "label")
    plt.figure(figsize=(10, 6))
    nx.draw(dg, pos, with_labels=False, node_size=2200)
    nx.draw_networkx_labels(dg, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.show()
