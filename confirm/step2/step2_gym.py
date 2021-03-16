from env.gym_barfem import BarFemGym


class Step2Gym(BarFemGym):
    """エッジを選択するのみの環境.といっても親クラスから変更はしていない．


    Args:
        BarFemGym ([type]): [description]
    """

    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes):
        super(Step2Gym, self).__init__(node_pos, input_nodes, input_vectors,
                                       output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes)
