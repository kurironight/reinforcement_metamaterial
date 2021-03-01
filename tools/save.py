import numpy as np
import pickle
import os


def save_graph_info(log_dir, nodes_positions, input_nodes, input_vectors,
                    output_nodes, output_vectors, frozen_nodes,
                    edges_indices, edges_thickness):
    """グラフの情報をlog_dir内部にnpy形式かpickle形式で保存する保存する．

    Args:
        log_dir (str)
        nodes_positions (np.array)
        input_nodes (list)
        input_vectors (np.array)
        output_nodes (list)
        output_vectors (np.array)
        frozen_nodes (list)
        edges_indices (np.array)
        edges_thickness (np.array)
    """
    os.makedirs(log_dir, exist_ok=True)
    np.save(os.path.join(log_dir, 'nodes_positions'),
            nodes_positions)
    np.save(os.path.join(log_dir, 'input_vectors'),
            input_vectors)
    np.save(os.path.join(log_dir, 'output_vectors'),
            output_vectors)
    np.save(os.path.join(log_dir, 'edges_indices'),
            edges_indices)
    np.save(os.path.join(log_dir, 'edges_thickness'),
            edges_thickness)
    with open(os.path.join(log_dir, 'input_nodes.pkl'), 'wb') as f:
        pickle.dump(input_nodes, f)
    with open(os.path.join(log_dir, 'output_nodes.pkl'), 'wb') as f:
        pickle.dump(output_nodes, f)
    with open(os.path.join(log_dir, 'frozen_nodes.pkl'), 'wb') as f:
        pickle.dump(frozen_nodes, f)
