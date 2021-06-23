import numpy as np


def make_edge_thick_triu_matrix(gene_edges_thickness, node_num, condition_edges_indices, condition_edges_thickness, edges_indices):
    """edge_thicknessを示す遺伝子から，condition_edge_thicknessを基にedges_thicknessを作成する関数
    """
    tri = np.zeros((node_num, node_num))
    tri[np.triu_indices(node_num, 1)] = gene_edges_thickness
    tri[(condition_edges_indices[:, 0], condition_edges_indices[:, 1])] = condition_edges_thickness
    edges_thickness = tri[(edges_indices[:, 0], edges_indices[:, 1])]
    return edges_thickness


def make_adj_triu_matrix(adj_element, node_num, condition_edges_indices):
    """隣接情報を示す遺伝子から，edge_indicesを作成する関数
    """
    adj_matrix = np.zeros((node_num, node_num))
    adj_matrix[np.triu_indices(node_num, 1)] = adj_element

    adj_matrix[(condition_edges_indices[:, 0], condition_edges_indices[:, 1])] = 1
    edge_indices = np.stack(np.where(adj_matrix), axis=1)

    return edge_indices
