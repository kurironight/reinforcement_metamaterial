import numpy as np
import gym
from tools.graph import convert_edge_indices_to_adj
from metamech.lattice import Lattice, EdgeInfoLattice
from metamech.actuator import Actuator
from metamech.viz import show_actuator
import networkx as nx
import os

gym.logger.set_level(40)

MAX_NODE = 100
LINEAR_STIFFNESS = 10
ANGULAR_STIFFNESS = 0.2
MAX_EDGE_THICKNESS = 5


class MetamechGym(gym.Env):
    # 定数定義

    # 初期化
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes,
                 output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes):
        super(MetamechGym, self).__init__()

        assert np.unique(node_pos, axis=0).shape[0] == node_pos.shape[0], "nodes_pos内部で同一の座標が存在してはいけない"
        edges_indices = np.sort(edges_indices, axis=1)  # edges_indicesの内部の順番を，[0,1],[3,7]のように，二番目のindexの方が値が大きいようにする
        assert np.unique(edges_indices, axis=0).shape[0] == edges_indices.shape[0], "edges_indices内部で同一のものが存在してはいけない"

        # 初期条件の指定
        self.max_node = MAX_NODE  # ノードの最大数

        self.first_node_pos = node_pos.copy()
        #
        self.condition_nodes = condition_nodes  # エッジの太さを変更しない条件ノードを指定する．
        self.input_nodes = input_nodes
        self.input_vectors = input_vectors
        self.output_nodes = output_nodes
        self.output_vectors = output_vectors
        self.frozen_nodes = frozen_nodes
        self.first_edges_indices = edges_indices.copy()
        self.first_edges_thickness = edges_thickness.copy()

        # current_status
        self.current_obs = {}

        # 行動空間と状態空間の定義
        self.action_space = gym.spaces.Dict({
            'new_node': gym.spaces.Box(low=0, high=1.0, shape=(1, 2),
                                       dtype=np.float32),
            'edge_thickness': gym.spaces.Box(low=np.array([-1]), high=np.array([1.0]), dtype=np.float32),
            'which_node': gym.spaces.MultiDiscrete([self.max_node - 1,
                                                    self.max_node]),
            'end': gym.spaces.Discrete(2),
        })

        self.observation_space = gym.spaces.Dict({
            # -1のところは，意味のないノードの情報
            'nodes': gym.spaces.Box(low=-1, high=1.0, shape=(self.max_node, 2), dtype=np.float32),
            'edges': gym.spaces.Dict({
                'adj': gym.spaces.MultiBinary([self.max_node, self.max_node]),
                # -1のところは，意味の無いエッジの情報
                'thickness': gym.spaces.Box(low=-1, high=1.0,
                                            shape=(self.max_node * self.max_node, 1), dtype=np.float32)
            })
        })

        self.info = {}  # edges_indicesのthicknessに応じた順序を保持したものを用意する．

    # 環境のリセット

    def reset(self):
        self._renew_current_obs(
            self.first_node_pos, self.first_edges_indices, self.first_edges_thickness)

        return self.current_obs

    def random_action(self):
        """強化学習を用いない場合に確認するための方針

        Returns:
            action
        """
        action = self.action_space.sample()

        # padding部分を排除した情報を抽出
        nodes_pos, adj, edges_thickness = self._extract_non_padding_status_from_current_obs()
        node_num = nodes_pos.shape[0]

        action['which_node'][0] = np.random.choice(np.arange(node_num))
        action['which_node'][1] = np.random.choice(
            np.delete(np.arange(node_num + 1), action['which_node'][0]))

        return action

    def step(self, action):

        # padding部分を排除した情報を抽出
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()
        node_num = nodes_pos.shape[0]
        if action['which_node'][1] < action['which_node'][0]:  # action["which_node"]の順番を正す
            ref = action['which_node'][1]
            action['which_node'][1] = action['which_node'][0]
            action['which_node'][0] = ref
        assert action['which_node'][1] <= node_num and action['which_node'][
            0] < node_num, 'action selects node which is higher than existing {} node or new node'.format(node_num)
        assert action['which_node'][1] != action['which_node'][0], 'same node are selected for action'

        if action['end']:  # 終了条件を満たす場合
            # TODO 本来はこれは，外側の方で行うこと
            reward = 1
            obs = self.current_obs
            self.info['status'] = 0
            return obs, reward, True, self.info

        self.info['status'] = 3  # 新規ノードを追加しないでエッジを追加する場合
        # 通常ルート
        if action['which_node'][1] == node_num:  # 新規ノードを追加する場合
            identical_node_index = np.asarray(np.where(nodes_pos[:, 0] == action['new_node'][0][0], True, False) & np.where(nodes_pos[:, 1] == action['new_node'][0][1], True, False)).nonzero()[0]
            if identical_node_index.shape[0] == 0:
                nodes_pos = np.concatenate([nodes_pos, action['new_node']])
                self.info['status'] = 4
            else:
                # 新規ノードと既に存在するノードとで一致するものがある場合
                action['which_node'][1] = identical_node_index
                if action['which_node'][0] == action['which_node'][1]:  # 追加するエッジのindiceが[1,1]や[2,2]などになってしまった場合
                    self.info['status'] = 5
                    return self.current_obs, 0, False, self.info

        # 既に存在するエッジを指定している場合
        index = np.arange(edges_indices.shape[0])
        index = index[np.isin(edges_indices[:, 0], action['which_node']) &
                      np.isin(edges_indices[:, 1], action['which_node'])]
        if index.shape != (0,):
            assert index.shape[0] == 1, 'there are two or more edge_indices which is identical'
            # もし，条件ノード間のエッジを選択した場合，何もしない
            ref_edge_indice = edges_indices[index][0]

            if np.isin([ref_edge_indice[0]], self.condition_nodes)[0] & np.isin([ref_edge_indice[1]], self.condition_nodes)[0]:
                # 条件ノード間のエッジを指定した場合
                self.info['status'] = 1
                return self.current_obs, 0, False, self.info
            else:
                # 条件ノード間のエッジ以外を選択した場合，そのエッジの太さを交換する
                edges_thickness[index] = action['edge_thickness']
                # renew obs
                self._renew_current_obs(
                    nodes_pos, edges_indices, edges_thickness)
                reward = 0
                obs = self.current_obs
                self.info['status'] = 2
            return obs, reward, False, self.info

        if action['which_node'][1] < action['which_node'][0]:  # action["which_node"]の順番を正す
            ref = action['which_node'][1]
            action['which_node'][1] = action['which_node'][0]
            action['which_node'][0] = ref
        edges_indices = np.concatenate([edges_indices, np.array(
            [[action['which_node'][0], action['which_node'][1]]])])

        edges_thickness = np.concatenate(
            [edges_thickness, action['edge_thickness']])

        self._renew_current_obs(nodes_pos, edges_indices, edges_thickness)

        reward = 0

        return self.current_obs, reward, False, self.info

    def confirm_graph_is_connected(self):
        # グラフが全て接続しているか確認

        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(nodes_pos)))
        G.add_edges_from(edges_indices)

        return nx.is_connected(G)

    def extract_node_edge_info(self):
        """グラフを構成するのに必要な情報を抽出

        Returns:
            nodes_pos[np.ndarray]: (*,2)
            edges_indices[np.ndarray]: (*,2)
            edges_thickness[np.ndarray]: (*,)
            adj[np.ndarray]: (*,*)
        """
        nodes_pos, adj, edges_thickness = self._extract_non_padding_status_from_current_obs()
        edges_indices = self.info['edges']['indices']
        return nodes_pos, edges_indices, edges_thickness, adj

    def calculate_simulation(self):
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        lattice = Lattice(
            nodes_positions=nodes_pos,
            edges_indices=edges_indices,
            edges_thickness=MAX_EDGE_THICKNESS * edges_thickness,
            linear_stiffness=LINEAR_STIFFNESS,
            angular_stiffness=ANGULAR_STIFFNESS
        )

        for edge in lattice._possible_edges:
            lattice.flip_edge(edge)

        actuator = Actuator(
            lattice=lattice,
            input_nodes=self.input_nodes,
            input_vectors=self.input_vectors,
            output_nodes=self.output_nodes,
            output_vectors=self.output_vectors,
            frozen_nodes=self.frozen_nodes
        )

        return actuator.efficiency

    # 環境の描画
    def render(self, save_path="image.png"):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        lattice = Lattice(
            nodes_positions=nodes_pos,
            edges_indices=edges_indices,
            edges_thickness=edges_thickness,
            linear_stiffness=LINEAR_STIFFNESS,
            angular_stiffness=ANGULAR_STIFFNESS
        )

        for edge in lattice._possible_edges:
            lattice.flip_edge(edge)

        actuator = Actuator(
            lattice=lattice,
            input_nodes=self.input_nodes,
            input_vectors=self.input_vectors,
            output_nodes=self.output_nodes,
            output_vectors=self.output_vectors,
            frozen_nodes=self.frozen_nodes
        )
        show_actuator(actuator, save_path=save_path)

    def _extract_non_padding_status_from_current_obs(self):
        """self.current_obsのうち，PADDINGを除いた部分を抽出
        """
        nodes_mask = self.current_obs['nodes'][:, 0] != -1  # 意味を成さない部分を除外
        vaild_nodes = self.current_obs['nodes'][nodes_mask]

        # 意味を成さない部分を除外
        thickness_mask = self.current_obs['edges']['thickness'] != -1
        vaild_edges_thickness = self.current_obs['edges']['thickness'][thickness_mask]

        node_num = vaild_nodes.shape[0]
        valid_adj = self.current_obs['edges']['adj'][:node_num, :node_num]

        return vaild_nodes, valid_adj, vaild_edges_thickness

    def _renew_current_obs(self, node_pos, edges_indices, edges_thickness):
        self.current_obs['nodes'] = np.pad(
            node_pos, ((0, self.max_node - node_pos.shape[0]), (0, 0)), constant_values=-1)
        adj = convert_edge_indices_to_adj(
            edges_indices, size=self.max_node)
        self.current_obs['edges'] = {
            'adj': adj,
            'thickness': np.pad(
                edges_thickness, (0, self.max_node * self.max_node - edges_thickness.shape[0]), constant_values=-1)}
        self.info['edges'] = {
            'indices': edges_indices,
        }


class EdgeInfoMetamechGym(MetamechGym):
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes,
                 output_vectors, frozen_nodes, edges_indices, edges_thickness):
        super().__init__(node_pos, input_nodes, input_vectors, output_nodes,
                         output_vectors, frozen_nodes, edges_indices, edges_thickness)

    def calculate_simulation(self):
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        lattice = EdgeInfoLattice(
            nodes_positions=nodes_pos,
            edges_indices=edges_indices,
            edges_thickness=edges_thickness,
            linear_stiffness=LINEAR_STIFFNESS,
            angular_stiffness=ANGULAR_STIFFNESS
        )

        for edge in lattice._possible_edges:
            lattice.flip_edge(edge)

        actuator = Actuator(
            lattice=lattice,
            input_nodes=self.input_nodes,
            input_vectors=self.input_vectors,
            output_nodes=self.output_nodes,
            output_vectors=self.output_vectors,
            frozen_nodes=self.frozen_nodes
        )

        return actuator.efficiency

    # 環境の描画
    def render(self, save_path="image.png"):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        lattice = EdgeInfoLattice(
            nodes_positions=nodes_pos,
            edges_indices=edges_indices,
            edges_thickness=edges_thickness,
            linear_stiffness=LINEAR_STIFFNESS,
            angular_stiffness=ANGULAR_STIFFNESS
        )

        for edge in lattice._possible_edges:
            lattice.flip_edge(edge)

        actuator = Actuator(
            lattice=lattice,
            input_nodes=self.input_nodes,
            input_vectors=self.input_vectors,
            output_nodes=self.output_nodes,
            output_vectors=self.output_vectors,
            frozen_nodes=self.frozen_nodes
        )
        show_actuator(actuator, save_path=save_path)
