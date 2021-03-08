from env.gym_barfem import BarFemGym


class Step1Gym(BarFemGym):
    """一つの要素の幅を変更するのみの環境.といっても親クラスから変更はしていない．


    Args:
        BarFemGym ([type]): [description]
    """

    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes):
        super(Step1Gym, self).__init__(node_pos, input_nodes, input_vectors,
                                       output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes)
