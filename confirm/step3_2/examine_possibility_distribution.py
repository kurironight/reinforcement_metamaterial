import pickle
import numpy as np


def examine_possibility_distribution(history):
    """A＞０，a_mean<aの時などの各事象の発生率を見る

    Returns:
        A>0&a_mean>a，A>0&a_mean<a,A<0&a_mean<a,A<0&a_mean>aが出現する確率を出力
    """
    data_num = len(history['epoch'])
    A_pos_a_small_to_mean = (np.array(history["advantage"]) >= 0) & (np.array(history["a"]) < np.array(history['a_mean']))
    A_pos_a_big_to_mean = (np.array(history["advantage"]) >= 0) & (np.array(history["a"]) >= np.array(history['a_mean']))
    A_neg_a_big_to_mean = (np.array(history["advantage"]) < 0) & (np.array(history["a"]) >= np.array(history['a_mean']))
    A_neg_a_small_to_mean = (np.array(history["advantage"]) < 0) & (np.array(history["a"]) < np.array(history['a_mean']))
    A_pos_a_small_to_mean_rate = np.count_nonzero(A_pos_a_small_to_mean) / data_num
    A_pos_a_big_to_mean_rate = np.count_nonzero(A_pos_a_big_to_mean) / data_num
    A_neg_a_big_to_mean_rate = np.count_nonzero(A_neg_a_big_to_mean) / data_num
    A_neg_a_small_to_mean_rate = np.count_nonzero(A_neg_a_small_to_mean) / data_num

    return A_pos_a_small_to_mean_rate, A_pos_a_big_to_mean_rate, A_neg_a_big_to_mean_rate, A_neg_a_small_to_mean_rate


def examine_pattern1_possibility_distribution(history):
    """パターン１発生時のa_mean<aの時などの各事象の発生率を見る

    Returns:
        A>0&a_mean>a，A>0&a_mean<aが出現する確率（母数は全体のepoch数-1）を出力
    """
    data_num = len(history['epoch'])  # 本来なら-1すべきかもしれないが，母数を合わせる為にそのままにする
    future_history_a_mean = np.array(history['a_mean'][1:])
    current_history_a_mean = np.array(history['a_mean'][:-1])
    current_history_advantage = np.array(history['advantage'][:-1])
    current_history_a = np.array(history['a'][:-1])

    A_pos_a_small_to_mean = ((current_history_advantage >= 0) & (current_history_a < current_history_a_mean)) & (future_history_a_mean < current_history_a_mean)
    A_pos_a_big_to_mean = ((current_history_advantage >= 0) & (current_history_a >= current_history_a_mean)) & (future_history_a_mean >= current_history_a_mean)
    A_pos_a_small_to_mean_rate = np.count_nonzero(A_pos_a_small_to_mean) / data_num
    A_pos_a_big_to_mean_rate = np.count_nonzero(A_pos_a_big_to_mean) / data_num

    return A_pos_a_small_to_mean_rate, A_pos_a_big_to_mean_rate


def examine_pattern2_possibility_distribution(history):
    """パターン２発生時のa_mean<aの時などの各事象の発生率を見る

    Returns:
        A>0&a_mean>a，A>0&a_mean<aが出現する確率（母数は全体のepoch数-1）を出力
    """

    with open("confirm/step3_2/Vp_edgethick_set/edges_thicknesses.pkl", 'rb') as web:
        edge_thicknesses = pickle.load(web)
    with open("confirm/step3_2/Vp_edgethick_set/rewards.pkl", 'rb') as web:
        rewards = pickle.load(web)
        rewards = np.array(rewards).flatten()

    Vp_a = np.array([convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp) for Vp in np.array(history['critic_value'])])
    future_Vp_a = Vp_a[1:]

    future_a_mean = np.array(history['a_mean'][1:])
    future_a = np.array(history['a'][1:])
    future_Vp = np.array(history['critic_value'][1:])
    future_Vp_a = np.array([convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp) for Vp in future_Vp])
    current_a_mean = np.array(history['a_mean'][:-1])
    current_Vp = np.array(history['critic_value'][:-1])
    current_Vp_a = np.array([convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp) for Vp in current_Vp])

    pattern2_occur = ((current_Vp_a - current_a_mean) * (current_a_mean - future_Vp_a)) > 0
    pattern2_data_num = np.count_nonzero(pattern2_occur)

    pattern2_small_direction = ((current_Vp_a - current_a_mean) > 0) * ((current_a_mean - future_Vp_a) > 0)
    pattern2_big_direction = ((current_Vp_a - current_a_mean) < 0) * ((current_a_mean - future_Vp_a) < 0)

    future_a_small_to_mean = (future_a_mean >= future_a)
    future_a_big_to_mean = (future_a_mean < future_a)

    big_dir_a_small_to_mean = pattern2_big_direction & future_a_small_to_mean
    big_dir_a_big_to_mean = pattern2_big_direction & future_a_big_to_mean
    small_dir_a_small_to_mean = pattern2_small_direction & future_a_small_to_mean
    small_dir_a_big_to_mean = pattern2_small_direction & future_a_big_to_mean

    big_dir_a_small_to_mean_rate = np.count_nonzero(big_dir_a_small_to_mean) / pattern2_data_num
    big_dir_a_big_to_mean_rate = np.count_nonzero(big_dir_a_big_to_mean) / pattern2_data_num
    small_dir_a_small_to_mean_rate = np.count_nonzero(small_dir_a_small_to_mean) / pattern2_data_num
    small_dir_a_big_to_mean_rate = np.count_nonzero(small_dir_a_big_to_mean) / pattern2_data_num

    return big_dir_a_small_to_mean_rate, big_dir_a_big_to_mean_rate, small_dir_a_small_to_mean_rate, small_dir_a_big_to_mean_rate


def convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp):
    """Vpから，エッジの太さを推測する
    """
    if Vp < 0:
        edge_thick = 0.5
    else:
        edgethick_index = np.searchsorted(rewards, Vp, side="left")  # 二分探索を行う
        edge_thick = edge_thicknesses[edgethick_index - 1]

    return edge_thick


def examine_pattern3_possibility_distribution(history, a_sigma_thresh=6.046791718933308 * 10**(-9)):
    data_num = len(history['epoch'])  # 本来なら-1すべきかもしれないが，母数を合わせる為にそのままにする
    current_history_a_sigma = np.array(history['a_sigma'][:-1])

    current_history_a_mean = np.array(history['a_mean'][:-1])
    current_history_advantage = np.array(history['advantage'][:-1])
    current_history_a = np.array(history['a'][:-1])

    pattern3_occur = current_history_a_sigma < a_sigma_thresh
    pattern3_data_num = np.count_nonzero(pattern3_occur)

    A_pos_a_small_to_mean = pattern3_occur & ((current_history_advantage >= 0) & (current_history_a < current_history_a_mean))
    A_pos_a_big_to_mean = pattern3_occur & ((current_history_advantage >= 0) & (current_history_a >= current_history_a_mean))
    A_neg_a_small_to_mean = pattern3_occur & ((current_history_advantage < 0) & (current_history_a < current_history_a_mean))
    A_neg_a_big_to_mean = pattern3_occur & ((current_history_advantage < 0) & (current_history_a >= current_history_a_mean))

    A_pos_a_small_to_mean_rate = np.count_nonzero(A_pos_a_small_to_mean) / pattern3_data_num
    A_pos_a_big_to_mean_rate = np.count_nonzero(A_pos_a_big_to_mean) / pattern3_data_num
    A_neg_a_small_to_mean_rate = np.count_nonzero(A_neg_a_small_to_mean) / pattern3_data_num
    A_neg_a_big_to_mean_rate = np.count_nonzero(A_neg_a_big_to_mean) / pattern3_data_num

    pattern3_occur_rate = pattern3_data_num / data_num

    return pattern3_occur_rate, A_pos_a_small_to_mean_rate, A_pos_a_big_to_mean_rate, A_neg_a_small_to_mean_rate, A_neg_a_big_to_mean_rate


with open("confirm/step3_2/a_gcn_c_gcn_results/50000確認/history.pkl", 'rb') as web:
    history = pickle.load(web)

print(examine_pattern3_possibility_distribution(history["2"]))
