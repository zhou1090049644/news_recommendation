import argparse
import math
import os
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate, get_mode_dir, is_eval_mode

warnings.filterwarnings('ignore')

random.seed(2020)

parser = argparse.ArgumentParser(description='hot recall')
parser.add_argument('--mode', default='valid')
parser.add_argument('--topk', default=50, type=int)
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
topk = args.topk
logfile = args.logfile

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'hot recall, mode: {mode}, topk: {topk}')


HOT_WEIGHTS = {
    'global_hot': 0.15,
    'category_hot': 0.35,
    'time_hot': 0.30,
    'context_hot': 0.20,
}

SCAN_LIMITS = {
    'global_hot': 200,
    'category_hot': 200,
    'time_hot': 300,
    'context_hot': 100,
}

DEFAULT_WINDOWS = [
    ('1d', 24 * 3600, 0.50),
    ('3d', 3 * 24 * 3600, 0.30),
    ('7d', 7 * 24 * 3600, 0.20),
]

DEFAULT_BUCKET_SECONDS = 3600

CONTEXT_GROUPS = [
    ('country_region_device_hour',
     ['click_country', 'click_region', 'click_deviceGroup', 'hour']),
    ('country_region', ['click_country', 'click_region']),
    ('device_hour', ['click_deviceGroup', 'hour']),
    ('country', ['click_country']),
]

CONTEXT_LEVEL_WEIGHTS = {
    'country_region_device_hour': 0.40,
    'country_region': 0.30,
    'device_hour': 0.20,
    'country': 0.10,
}


def _normalize_timestamp(s):
    s = pd.to_numeric(s, errors='coerce')
    if s.dropna().empty:
        return s

    # Tianchi click_timestamp and created_at_ts are millisecond timestamps.
    if s.dropna().median() > 10**11:
        return s / 1000
    return s


def _prepare_click(df_click, df_article=None):
    df = df_click.copy()
    if 'click_article_id' in df.columns:
        df = df.rename(columns={'click_article_id': 'article_id'})

    df['click_timestamp'] = _normalize_timestamp(df['click_timestamp'])

    if df_article is not None:
        article_cols = ['article_id']
        for col in ['category_id', 'created_at_ts']:
            if col in df_article.columns:
                article_cols.append(col)
        df = df.merge(df_article[article_cols], how='left', on='article_id')

    if 'created_at_ts' in df.columns:
        df['created_at_ts'] = _normalize_timestamp(df['created_at_ts'])

    df['hour'] = pd.to_datetime(df['click_timestamp'], unit='s').dt.hour

    return df.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)


def _prepare_article(df_article):
    df = df_article.copy()
    if 'created_at_ts' in df.columns:
        df['created_at_ts'] = _normalize_timestamp(df['created_at_ts'])
    return df


def _rank_score(rank):
    return 1.0 / (rank + 1)


def _time_bucket(ts, bucket_seconds=DEFAULT_BUCKET_SECONDS):
    if pd.isna(ts):
        return None
    return int(ts // bucket_seconds * bucket_seconds)


def _hot_list(df, group_cols=None, now_ts=None, tau=None, topn=500):
    if now_ts is None:
        now_ts = df['click_timestamp'].max()
    if tau is None:
        tau = 7 * 24 * 3600

    keys = [] if group_cols is None else list(group_cols)
    stat = df.groupby(keys + ['article_id']).agg(
        cnt=('article_id', 'size'),
        last_ts=('click_timestamp', 'max'),
    ).reset_index()
    stat['hot_score'] = np.log1p(stat['cnt']) * np.exp(
        -(now_ts - stat['last_ts']) / tau)

    if not keys:
        stat = stat.sort_values('hot_score', ascending=False).head(topn)
        return list(zip(stat['article_id'].astype(int), stat['hot_score']))

    ans = {}
    stat = stat.sort_values(keys + ['hot_score'],
                            ascending=[True] * len(keys) + [False])
    for group_key, g in stat.groupby(keys):
        if not isinstance(group_key, tuple):
            group_key = (group_key, )
        ans[group_key] = list(
            zip(g.head(topn)['article_id'].astype(int), g.head(topn)['hot_score']))
    return ans


def _history_by_bucket(df, buckets=None, bucket_seconds=DEFAULT_BUCKET_SECONDS):
    df = df.sort_values('click_timestamp').reset_index(drop=True)
    if buckets is None:
        df['time_bucket'] = df['click_timestamp'].apply(
            lambda x: _time_bucket(x, bucket_seconds))
        buckets = sorted(df['time_bucket'].dropna().unique())
    else:
        buckets = sorted(x for x in buckets if x is not None)

    click_ts = df['click_timestamp'].values
    for bucket_ts in tqdm(buckets):
        end = np.searchsorted(click_ts, bucket_ts, side='right')
        yield int(bucket_ts), df.iloc[:end]


def _get_bucket_hot(hot_by_bucket, query_time):
    if not hot_by_bucket:
        return None

    bucket_ts = _time_bucket(query_time)
    if bucket_ts in hot_by_bucket:
        return hot_by_bucket[bucket_ts]

    valid_buckets = [x for x in hot_by_bucket.keys() if x <= bucket_ts]
    if not valid_buckets:
        return hot_by_bucket[min(hot_by_bucket.keys())]
    return hot_by_bucket[max(valid_buckets)]


def build_global_hot_from_prepared(df_click_prepared, buckets=None):
    df = df_click_prepared
    global_hot_by_bucket = {}
    for bucket_ts, df_hist in _history_by_bucket(df, buckets=buckets):
        if df_hist.empty:
            continue
        global_hot_by_bucket[bucket_ts] = _hot_list(df_hist,
                                                    now_ts=bucket_ts,
                                                    tau=7 * 24 * 3600,
                                                    topn=1000)
    return global_hot_by_bucket


def build_global_hot(df_click, df_article):
    df = _prepare_click(df_click, df_article)
    return build_global_hot_from_prepared(df)


def build_category_hot_from_prepared(df_click_prepared, buckets=None):
    df = df_click_prepared
    if 'category_id' not in df.columns:
        return {}

    category_hot_by_bucket = {}
    for bucket_ts, df_hist in _history_by_bucket(df, buckets=buckets):
        if df_hist.empty:
            continue
        category_hot_by_bucket[bucket_ts] = _hot_list(df_hist,
                                                      group_cols=[
                                                          'category_id'
                                                      ],
                                                      now_ts=bucket_ts,
                                                      tau=7 * 24 * 3600,
                                                      topn=500)
    return category_hot_by_bucket


def build_category_hot(df_click, df_article):
    df = _prepare_click(df_click, df_article)
    return build_category_hot_from_prepared(df)


def build_time_hot_from_prepared(df_click_prepared, windows, buckets=None):
    df = df_click_prepared
    time_hot_by_bucket = {}
    for bucket_ts, df_hist in _history_by_bucket(df, buckets=buckets):
        item_score = defaultdict(float)
        for _, seconds, weight in windows:
            start_ts = bucket_ts - seconds
            df_win = df_hist[df_hist['click_timestamp'] >= start_ts]
            if df_win.empty:
                continue

            win_hot = _hot_list(df_win,
                                now_ts=bucket_ts,
                                tau=seconds,
                                topn=1000)
            for rank, (article_id, _) in enumerate(win_hot):
                item_score[int(article_id)] += weight * _rank_score(rank)

        time_hot_by_bucket[bucket_ts] = sorted(item_score.items(),
                                               key=lambda x: x[1],
                                               reverse=True)
    return time_hot_by_bucket


def build_time_hot(df_click, df_article, windows):
    df = _prepare_click(df_click, df_article)
    return build_time_hot_from_prepared(df, windows)


def build_context_hot_from_prepared(df_click_prepared, buckets=None):
    df = df_click_prepared
    context_hot_by_bucket = {}
    for bucket_ts, df_hist in _history_by_bucket(df, buckets=buckets):
        if df_hist.empty:
            continue

        context_hot = {}
        for name, cols in CONTEXT_GROUPS:
            valid_cols = [col for col in cols if col in df_hist.columns]
            if len(valid_cols) != len(cols):
                continue
            context_hot[name] = _hot_list(df_hist,
                                          group_cols=cols,
                                          now_ts=bucket_ts,
                                          tau=3 * 24 * 3600,
                                          topn=300)
        context_hot_by_bucket[bucket_ts] = context_hot
    return context_hot_by_bucket


def build_context_hot(df_click, df_article):
    df = _prepare_click(df_click, df_article)
    return build_context_hot_from_prepared(df)


def get_user_profile_from_prepared(df_click_prepared):
    df = df_click_prepared
    profiles = {}
    for user_id, g in tqdm(df.groupby('user_id')):
        g = g.sort_values('click_timestamp')
        last_row = g.iloc[-1]

        cat_pref = defaultdict(float)
        if 'category_id' in g.columns:
            recent = g.tail(5)
            n = recent.shape[0]
            for i, (_, row) in enumerate(recent.iterrows()):
                if pd.isna(row['category_id']):
                    continue
                # Newer clicks get larger weights.
                cat_pref[row['category_id']] += 0.8**(n - i - 1)

        context = {}
        for col in [
                'click_country', 'click_region', 'click_deviceGroup', 'hour'
        ]:
            if col in g.columns and not pd.isna(last_row[col]):
                context[col] = last_row[col]

        profiles[user_id] = {
            'clicked_set': set(g['article_id'].astype(int)),
            'query_time': last_row['click_timestamp'],
            'category_pref': dict(cat_pref),
            'context': context,
        }

    return profiles


def get_user_profile(df_click, df_article):
    df = _prepare_click(df_click, df_article)
    return get_user_profile_from_prepared(df)


def _query_buckets(df_query, df_click_prepared):
    query_users = set(df_query['user_id'].unique())
    user_query_time = df_click_prepared[df_click_prepared['user_id'].isin(
        query_users)].groupby('user_id')['click_timestamp'].max()
    buckets = {
        _time_bucket(query_time)
        for query_time in user_query_time.values
        if not pd.isna(query_time)
    }

    if not buckets and not df_click_prepared.empty:
        buckets.add(_time_bucket(df_click_prepared['click_timestamp'].max()))
    return buckets


def _can_recall(article_id, user_profile, article_created_map):
    if article_id in user_profile['clicked_set']:
        return False

    created_at = article_created_map.get(article_id)
    if created_at is not None and not pd.isna(created_at):
        if created_at > user_profile['query_time']:
            return False

    return True


def _add_candidate(candidates, article_id, recall_type, score):
    article_id = int(article_id)
    candidates[article_id]['score'] += score
    candidates[article_id]['type_scores'][recall_type] += score


def _context_keys(user_profile):
    ctx = user_profile.get('context', {})
    keys = []
    if all(col in ctx and not pd.isna(ctx[col]) for col in
           ['click_country', 'click_region', 'click_deviceGroup', 'hour']):
        keys.append(('country_region_device_hour',
                     (ctx['click_country'], ctx['click_region'],
                      ctx['click_deviceGroup'], ctx['hour'])))
    if all(col in ctx and not pd.isna(ctx[col])
           for col in ['click_country', 'click_region']):
        keys.append(('country_region',
                     (ctx['click_country'], ctx['click_region'])))
    if all(col in ctx and not pd.isna(ctx[col])
           for col in ['click_deviceGroup', 'hour']):
        keys.append(('device_hour', (ctx['click_deviceGroup'], ctx['hour'])))
    if 'click_country' in ctx and not pd.isna(ctx['click_country']):
        keys.append(('country', (ctx['click_country'], )))
    return keys


def recall_for_user(user_id, user_profile, hot_dicts, topk):
    article_created_map = hot_dicts.get('article_created_map', {})
    user_hot = {
        'global_hot':
        _get_bucket_hot(hot_dicts['global_hot'], user_profile['query_time'])
        or [],
        'category_hot':
        _get_bucket_hot(hot_dicts['category_hot'], user_profile['query_time'])
        or {},
        'time_hot':
        _get_bucket_hot(hot_dicts['time_hot'], user_profile['query_time'])
        or [],
        'context_hot':
        _get_bucket_hot(hot_dicts['context_hot'], user_profile['query_time'])
        or {},
    }
    candidates = defaultdict(lambda: {
        'score': 0.0,
        'type_scores': defaultdict(float)
    })

    for rank, (article_id, _) in enumerate(
            user_hot['global_hot'][:SCAN_LIMITS['global_hot']]):
        if _can_recall(article_id, user_profile, article_created_map):
            _add_candidate(candidates, article_id, 'global_hot',
                           HOT_WEIGHTS['global_hot'] * _rank_score(rank))

    category_pref = user_profile.get('category_pref', {})
    total_pref = sum(category_pref.values())
    if total_pref > 0:
        for category_id, pref in sorted(category_pref.items(),
                                       key=lambda x: x[1],
                                       reverse=True):
            category_items = user_hot['category_hot'].get(
                (category_id, ), [])[:SCAN_LIMITS['category_hot']]
            pref_weight = pref / total_pref
            for rank, (article_id, _) in enumerate(category_items):
                if _can_recall(article_id, user_profile, article_created_map):
                    _add_candidate(
                        candidates, article_id, 'category_hot',
                        HOT_WEIGHTS['category_hot'] * pref_weight *
                        _rank_score(rank))

    for rank, (article_id, _) in enumerate(
            user_hot['time_hot'][:SCAN_LIMITS['time_hot']]):
        if _can_recall(article_id, user_profile, article_created_map):
            _add_candidate(candidates, article_id, 'time_hot',
                           HOT_WEIGHTS['time_hot'] * _rank_score(rank))

    # Context recall is sparse. Try fine-grained keys first, then fallback
    # naturally to coarser groups. Level weights share one context budget.
    for group_name, group_key in _context_keys(user_profile):
        group_items = user_hot['context_hot'].get(group_name, {}).get(
            group_key, [])[:SCAN_LIMITS['context_hot']]
        level_weight = CONTEXT_LEVEL_WEIGHTS.get(group_name, 0)
        for rank, (article_id, _) in enumerate(group_items):
            if _can_recall(article_id, user_profile, article_created_map):
                _add_candidate(candidates, article_id, 'context_hot',
                               HOT_WEIGHTS['context_hot'] * level_weight *
                               _rank_score(rank))

    rows = []
    for article_id, info in candidates.items():
        recall_type = max(info['type_scores'].items(),
                          key=lambda x: x[1])[0]
        type_scores = info['type_scores']
        hot_recall_source_count = sum(1 for score in type_scores.values()
                                      if score > 0)
        rows.append(
            (user_id, article_id, info['score'], recall_type,
             type_scores.get('global_hot', 0.0),
             type_scores.get('category_hot', 0.0),
             type_scores.get('time_hot', 0.0),
             type_scores.get('context_hot', 0.0), hot_recall_source_count))

    rows = sorted(rows, key=lambda x: x[2], reverse=True)[:topk]
    return rows


def hot_recall(df_query, df_click, df_article, mode, topk=50):
    """
    Return hot recall result with columns:
    user_id, article_id, sim_score, recall_type, global_hot_score,
    category_hot_score, time_hot_score, context_hot_score,
    hot_recall_source_count
    """
    df_article = _prepare_article(df_article)
    article_created_map = {}
    if 'created_at_ts' in df_article.columns:
        article_created_map = dict(
            zip(df_article['article_id'].astype(int), df_article['created_at_ts']))

    df_click_prepared = _prepare_click(df_click, df_article)
    query_buckets = _query_buckets(df_query, df_click_prepared)

    hot_dicts = {
        'global_hot':
        build_global_hot_from_prepared(df_click_prepared,
                                       buckets=query_buckets),
        'category_hot':
        build_category_hot_from_prepared(df_click_prepared,
                                         buckets=query_buckets),
        'time_hot':
        build_time_hot_from_prepared(df_click_prepared,
                                     DEFAULT_WINDOWS,
                                     buckets=query_buckets),
        'context_hot':
        build_context_hot_from_prepared(df_click_prepared,
                                        buckets=query_buckets),
        'article_created_map': article_created_map,
    }

    user_profiles = get_user_profile_from_prepared(df_click_prepared)
    rows = []
    for user_id in tqdm(df_query['user_id'].unique()):
        if user_id not in user_profiles:
            # Pure cold-start users fall back to global hot. Use a loose profile
            # with no history and query time equal to training max time.
            user_profiles[user_id] = {
                'clicked_set': set(),
                'query_time': df_click_prepared['click_timestamp'].max(),
                'category_pref': {},
                'context': {},
            }
        rows.extend(
            recall_for_user(user_id, user_profiles[user_id], hot_dicts, topk))

    df_data = pd.DataFrame(rows,
                           columns=[
                               'user_id', 'article_id', 'sim_score',
                               'recall_type', 'global_hot_score',
                               'category_hot_score', 'time_hot_score',
                               'context_hot_score', 'hot_recall_source_count'
                           ])
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    df_data['user_id'] = df_data['user_id'].astype(int)
    df_data['article_id'] = df_data['article_id'].astype(int)
    df_data['hot_recall_source_count'] = df_data[
        'hot_recall_source_count'].astype(int)

    if mode in ['valid', 'testa_valid']:
        df_label = df_query.rename(columns={'click_article_id': 'article_id'})
        df_label = df_label[['user_id', 'article_id']].copy()
        df_label['label'] = 1
        df_data = df_data.merge(df_label,
                                on=['user_id', 'article_id'],
                                how='left')
        df_data['label'] = df_data['label'].fillna(0).astype(int)
    else:
        df_data['label'] = np.nan

    ordered_cols = [
        'user_id', 'article_id', 'sim_score', 'label', 'recall_type',
        'global_hot_score', 'category_hot_score', 'time_hot_score',
        'context_hot_score', 'hot_recall_source_count'
    ]
    df_data = df_data[ordered_cols]
    return df_data


if __name__ == '__main__':
    mode_dir = get_mode_dir(mode)
    df_click = pd.read_pickle(f'../user_data/data/{mode_dir}/click.pkl')
    df_query = pd.read_pickle(f'../user_data/data/{mode_dir}/query.pkl')
    df_article = pd.read_csv('../tcdata/articles.csv')

    df_data = hot_recall(df_query, df_click, df_article, mode, topk=topk)

    log.debug(f'recall_hot.shape: {df_data.shape}')
    log.debug(f'recall_hot.head: {df_data.head()}')

    if is_eval_mode(mode):
        log.info('calculate hot recall metrics')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'hot: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
        log.debug(f"hot label distribution: {df_data['label'].value_counts()}")

    os.makedirs(f'../user_data/data/{mode_dir}', exist_ok=True)
    df_data.to_pickle(f'../user_data/data/{mode_dir}/recall_hot.pkl')
