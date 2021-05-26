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


def examine_pattern1_possibility_distribution(history, epoch_start_end=None):
    """パターン１発生時のa_mean<aの時などの各事象の発生率を見る

    Args:
        history ([dictionary]): 学習の経緯を保存している辞書
        epoch_start_end ([tuple], optional): (start_epoch,end_epoch)という形式で，historyの内の調査範囲を指定することができる. Defaults to None.

    Returns:
        A>0&a_mean>a，A>0&a_mean<aが出現する確率（母数は全体のepoch数-1）を出力
    """
    if epoch_start_end is not None:
        epoch_range = list(range(epoch_start_end[0] - 1, epoch_start_end[1]))
    else:
        epoch_range = list(range(len(history['epoch'])))

    data_num = len(np.array(history['epoch'])[epoch_range])  # 本来なら-1すべきかもしれないが，母数を合わせる為にそのままにする
    future_history_a_mean = np.array(np.array(history['a_mean'])[epoch_range][1:])
    current_history_a_mean = np.array(np.array(history['a_mean'])[epoch_range][:-1])
    current_history_advantage = np.array(np.array(history['advantage'])[epoch_range][:-1])
    current_history_a = np.array(np.array(history['a'])[epoch_range][:-1])

    A_pos_a_small_to_mean = ((current_history_advantage >= 0) & (current_history_a < current_history_a_mean)) & (future_history_a_mean >= current_history_a_mean)
    A_pos_a_big_to_mean = ((current_history_advantage >= 0) & (current_history_a >= current_history_a_mean)) & (future_history_a_mean < current_history_a_mean)
    A_pos_a_small_to_mean_rate = np.count_nonzero(A_pos_a_small_to_mean) / data_num
    A_pos_a_big_to_mean_rate = np.count_nonzero(A_pos_a_big_to_mean) / data_num

    pattern1_occur_rate = A_pos_a_small_to_mean_rate + A_pos_a_big_to_mean_rate
    A_pos_a_small_to_mean_rate /= pattern1_occur_rate
    A_pos_a_big_to_mean_rate /= pattern1_occur_rate

    good_rate = A_pos_a_small_to_mean_rate * pattern1_occur_rate
    bad_rate = A_pos_a_big_to_mean_rate * pattern1_occur_rate

    return pattern1_occur_rate, A_pos_a_small_to_mean_rate, A_pos_a_big_to_mean_rate, good_rate, bad_rate


def examine_pattern2_possibility_distribution(history, epoch_start_end=None):
    """パターン２発生時のa_mean<aの時などの各事象の発生率を見る

    Returns:
        A>0&a_mean>a，A>0&a_mean<aが出現する確率（母数は全体のepoch数-1）を出力
    """

    with open("confirm/step3_2/Vp_edgethick_set/edges_thicknesses.pkl", 'rb') as web:
        edge_thicknesses = pickle.load(web)
    with open("confirm/step3_2/Vp_edgethick_set/rewards.pkl", 'rb') as web:
        rewards = pickle.load(web)
        rewards = np.array(rewards).flatten()

    if epoch_start_end is not None:
        epoch_range = list(range(epoch_start_end[0] - 1, epoch_start_end[1]))
    else:
        epoch_range = list(range(len(history['epoch'])))

    data_num = len(np.array(history['epoch'])[epoch_range])  # 本来なら-1すべきかもしれないが，母数を合わせる為にそのままにする
    Vp_a = np.array([convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp) for Vp in np.array(np.array(history['critic_value'])[epoch_range])])
    future_Vp_a = Vp_a[1:]
    current_Vp_a = Vp_a[:-1]

    future_a_mean = np.array(np.array(history['a_mean'])[epoch_range][1:])
    future_a = np.array(np.array(history['a'])[epoch_range][1:])
    current_a_mean = np.array(np.array(history['a_mean'])[epoch_range][:-1])

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

    pattern2_occur_rate = pattern2_data_num / data_num

    good_rate = (big_dir_a_big_to_mean_rate + small_dir_a_big_to_mean_rate) * pattern2_occur_rate
    bad_rate = (big_dir_a_small_to_mean_rate + small_dir_a_small_to_mean_rate) * pattern2_occur_rate

    return pattern2_occur_rate, big_dir_a_small_to_mean_rate, big_dir_a_big_to_mean_rate, small_dir_a_small_to_mean_rate, small_dir_a_big_to_mean_rate, good_rate, bad_rate


def convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp):
    """Vpから，エッジの太さを推測する
    """
    if Vp < 0:
        edge_thick = 0.5
    else:
        edgethick_index = np.searchsorted(rewards, Vp, side="left")  # 二分探索を行う
        edge_thick = edge_thicknesses[edgethick_index - 1]

    return edge_thick


def examine_pattern3_possibility_distribution(history, a_sigma_thresh=6.046791718933308 * 10**(-9), epoch_start_end=None):

    if epoch_start_end is not None:
        epoch_range = list(range(epoch_start_end[0] - 1, epoch_start_end[1]))
    else:
        epoch_range = list(range(len(history['epoch'])))

    data_num = len(np.array(history['epoch'])[epoch_range])  # 本来なら-1すべきかもしれないが，母数を合わせる為にそのままにする
    current_history_a_sigma = np.array(np.array(history['a_sigma'])[epoch_range][:-1])

    current_history_a_mean = np.array(np.array(history['a_mean'])[epoch_range][:-1])
    current_history_advantage = np.array(np.array(history['advantage'])[epoch_range][:-1])
    current_history_a = np.array(np.array(history['a'])[epoch_range][:-1])

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
    good_rate = (A_pos_a_big_to_mean_rate + A_neg_a_big_to_mean_rate) * pattern3_occur_rate
    bad_rate = (A_pos_a_small_to_mean_rate + A_neg_a_small_to_mean_rate) * pattern3_occur_rate

    return pattern3_occur_rate, A_pos_a_small_to_mean_rate, A_pos_a_big_to_mean_rate, A_neg_a_small_to_mean_rate, A_neg_a_big_to_mean_rate, good_rate, bad_rate


"""

with open("confirm/step3_2/a_gcn_c_gcn_results/50000確認/history.pkl", 'rb') as web:
    history = pickle.load(web)

max_converge_range = (1, 6250)
max_converge_range = (2001, 6250)
minimize_range = (6500, 11311)
min_converge_range = (20000, len(history["2"]['epoch']))


examine_pattern1_possibility_distribution(history["2"], epoch_start_end=max_converge_range)

print(examine_pattern2_possibility_distribution(history["2"], epoch_start_end=max_converge_range))

examine_pattern1_possibility_distribution(history["2"], epoch_start_end=minimize_range)


print(examine_pattern2_possibility_distribution(history["2"], epoch_start_end=minimize_range))

examine_pattern1_possibility_distribution(history["2"], epoch_start_end=min_converge_range)


print(examine_pattern2_possibility_distribution(history["2"], epoch_start_end=min_converge_range))
examine_pattern3_possibility_distribution(history["2"], epoch_start_end=min_converge_range)
"""
