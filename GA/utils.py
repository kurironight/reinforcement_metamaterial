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


def revert_edge_indices_to_binary(edges_indices, node_num):
    """edge_indicesから，もとの隣接情報を示すbinary情報に戻す関数
    """
    adj_matrix = np.zeros((node_num, node_num))
    adj_matrix[(edges_indices[:, 0], edges_indices[:, 1])] = 1
    binary_adj_element = np.array(adj_matrix[np.triu_indices(node_num, 1)]).astype(np.bool)

    return binary_adj_element


def add_edge_indices_to_gene_edge_indices(gene_edge_indices, additional_edge_indices, gene_node_num):
    """gene_edge_indices(バイナリ情報)に対し，新しいノードに対するエッジ情報（バイナリ情報）を適切に追加する関数
    """
    adj_matrix = np.zeros((gene_node_num, gene_node_num), dtype=bool)
    adj_matrix[np.triu_indices(gene_node_num, 1)] = np.array(gene_edge_indices).squeeze()
    additional_edge_indices = np.reshape(additional_edge_indices, [gene_node_num, 1])
    new_adj_matrix = np.concatenate([adj_matrix, additional_edge_indices], axis=1)
    new_adj_matrix = np.concatenate([new_adj_matrix, np.zeros((1, gene_node_num + 1), dtype=bool)], axis=0)
    new_gene_edge_indices = new_adj_matrix[np.triu_indices(gene_node_num + 1, 1)].reshape((-1, 1)).tolist()
    return new_gene_edge_indices
