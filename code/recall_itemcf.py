import argparse
import math
import os
import pickle
import platform
import random
import signal
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate, get_mode_dir, is_eval_mode

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('thread' if platform.system() == 'Windows' else 'process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    # {1001: [23, 45, 89],1002: [12, 45],1003: [77, 23, 90]}
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_cnt = defaultdict(int)
    sim_dict = {}

    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            # 用来存放“当前文章 item 和其他文章的相似度”
            sim_dict.setdefault(item, {})

            # 用来找“和当前文章 item 一起被同一个用户点击过的其他文章”
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))

    # sim_dict = {
    #     itemA: {
    #         itemB: score_AB,
    #         itemC: score_AC
    #     },
    #     itemB: {
    #         itemA: score_BA,
    #         itemC: score_BC
    #     }
    # }
    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        # 处理冷启动/无历史行为用户
        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                # 曝光过滤”或“历史去重
                if relate_item not in interacted_items:
                    # 保存每个候选物品的最终推荐得分
                    rank.setdefault(relate_item, 0)
                    # 推荐分数也要计算位置衰减权重
                    rank[relate_item] += wij * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    mode_dir = get_mode_dir(mode)
    df_click = pd.read_pickle(f'../user_data/data/{mode_dir}/click.pkl')
    df_query = pd.read_pickle(f'../user_data/data/{mode_dir}/query.pkl')

    os.makedirs(f'../user_data/sim/{mode_dir}', exist_ok=True)
    # 保存文章相似度字典
    sim_pkl_file = f'../user_data/sim/{mode_dir}/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if is_eval_mode(mode):
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    # 保存召回结果，字段大致是：user_id, article_id, sim_score, label   -- label:0, 1, NaN
    df_data.to_pickle(f'../user_data/data/{mode_dir}/recall_itemcf.pkl')
