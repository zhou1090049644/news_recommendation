import argparse
import math
import os
import pickle
import platform
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate, get_mode_dir, is_eval_mode

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('thread' if platform.system() == 'Windows' else 'process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    # sentences：
    # [
    #     [101, 102, 105],
    #     [102, 108],
    #     [201, 105, 306]
    # ]
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         size=256,          # 表示每篇文章会被表示成一个 256 维向量
                         window=3,          # 表示训练时关注当前文章前后 3 个位置内的文章
                         min_count=1,       # 表示即使某篇文章只出现过一次，也会参与训练
                         sg=1,              # 表示使用 Skip-gram 模型
                         hs=0,
                         seed=seed,
                         negative=5,        # 表示使用负采样，每个正样本搭配 5 个负样本，提高训练效率
                         workers=10,
                         iter=1)
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model:
            article_vec_map[int(word)] = model[word]

    # {
    #     101: array([0.12, -0.03, ..., 0.45]),
    #     102: array([-0.08, 0.21, ..., 0.11])
    # }
    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]

        for item in interacted_items:
            if item not in article_vec_map:
                continue

            article_vec = article_vec_map[item]

            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
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

    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    mode_dir = get_mode_dir(mode)
    df_click = pd.read_pickle(f'../user_data/data/{mode_dir}/click.pkl')
    df_query = pd.read_pickle(f'../user_data/data/{mode_dir}/query.pkl')

    os.makedirs(f'../user_data/data/{mode_dir}', exist_ok=True)
    os.makedirs(f'../user_data/model/{mode_dir}', exist_ok=True)

    w2v_file = f'../user_data/data/{mode_dir}/article_w2v.pkl'
    model_path = f'../user_data/model/{mode_dir}'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
    # for path, _, file_list in os.walk('../tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
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
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    df_data.to_pickle(f'../user_data/data/{mode_dir}/recall_w2v.pkl')
