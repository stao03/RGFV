# -*- coding: utf-8 -*-

import os
import json
import hashlib
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import pearsonr

import local_llm


ROOT = os.path.expanduser('~/kg/dataset/CHSIMS')

EGEMAPS = f'{ROOT}/features/trans_audio/egemaps.csv'
PYFEAT = f'{ROOT}/features/trans_video_8f/pyfeat_agg.csv'
LABELS = f'{ROOT}/label.csv'


CONDITIONS = {
    'T': ['audio', 'visual'],
    'A': ['text', 'visual'],
    'V': ['text', 'audio'],
    'TA': ['visual'],
    'TV': ['audio'],
    'AV': ['text'],
    'TAV': []
}


MAXN = None


# ============================================================
# Completion config
# ============================================================

# 统一的 Top-K 参数：
# - text embedding 检索用 TOPK
# - A/V 特征检索也用 TOPK
# - 补 audio / visual 时，对 top-k 邻居求均值
# - 补 text 时，拼接
TOPK = 3

# p/r 特征筛选阈值：p < P_THRESH 准入，再按 |r| 降序排序
P_THRESH = 0.3

KB_BINARY_ONLY = True

# 现在 T 也可以补，因为有 text embedding 检索
COMPLETION_CONDS = {'T', 'A', 'V', 'TA', 'TV', 'AV'}

# text embedding cache
CACHE_DIR = 'cache'
TEXT_EMB_CACHE = f'{CACHE_DIR}/kb_text_qwen2p5_7b_embeddings.npz'
TEXT_META_CACHE = f'{CACHE_DIR}/kb_text_qwen2p5_7b_meta.json'

FORCE_REBUILD_TEXT_EMB = False

# Qwen2.5-7B 提 hidden states 比较吃显存，先用 2 或 4
TEXT_EMB_BATCH_SIZE = 8
TEXT_EMB_MAX_LENGTH = 256


# ============================================================
# Read data
# ============================================================

LABEL_COLUMNS = [
    'video_id',
    'clip_id',
    'text',
    'label',
    'label_T',
    'label_A',
    'label_V',
    'annotation',
    'mode',
]


def read_labels(path):
    try:
        data = pd.read_csv(
            path,
            header=None,
            names=LABEL_COLUMNS,
            dtype={
                'video_id': 'string',
                'clip_id': 'string',
            },
            encoding='utf-8-sig',
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            path,
            header=None,
            names=LABEL_COLUMNS,
            dtype={
                'video_id': 'string',
                'clip_id': 'string',
            },
            encoding='gb18030',
        )

    # 兼容标签文件实际带有表头的情况。
    if (
        not data.empty
        and str(data.iloc[0]['video_id']).strip().lower() == 'video_id'
    ):
        data = data.iloc[1:].reset_index(drop=True)

    data['video_id'] = (
        data['video_id']
        .astype('string')
        .str.strip()
    )

    # 保留 clip_id 的前导零，并兼容 Excel 写出的 1.0。
    data['clip_id'] = (
        data['clip_id']
        .astype('string')
        .str.strip()
        .str.replace(r'\.0+$', '', regex=True)
        .str.zfill(4)
    )

    # 兼容 video_1、video_001 等写法，统一成 video_0001。
    video_number = data['video_id'].str.extract(
        r'^video_(\d+)$',
        expand=False,
    )
    video_mask = video_number.notna()
    data.loc[video_mask, 'video_id'] = (
        'video_' + video_number[video_mask].str.zfill(4)
    )

    data['text'] = data['text'].fillna('').astype(str)
    data['mode'] = (
        data['mode']
        .astype('string')
        .str.strip()
        .str.lower()
    )
    data['annotation'] = (
        data['annotation']
        .astype('string')
        .str.strip()
    )

    for column in ['label', 'label_T', 'label_A', 'label_V']:
        data[column] = pd.to_numeric(
            data[column],
            errors='coerce',
        )

    # video_0001 + 0001 -> video_0001_0001
    data['segment_id'] = data['video_id'] + '_' + data['clip_id']

    return data


def read_features(path, feature_name):
    data = pd.read_csv(path, encoding='utf-8-sig')
    data.columns = data.columns.astype(str).str.strip()

    if 'segment_id' not in data.columns:
        raise ValueError(
            f'{feature_name} 中没有 segment_id 列，'
            f'当前前十列为：{data.columns.tolist()[:10]}'
        )

    data['segment_id'] = (
        data['segment_id']
        .astype('string')
        .str.strip()
    )

    return data


ege = read_features(EGEMAPS, 'eGeMAPS audio features')
pf = read_features(PYFEAT, 'PyFeat visual features')
lab = read_labels(LABELS)

for frame_name, frame in [
    ('labels', lab),
    ('visual features', pf),
    ('audio features', ege),
]:
    duplicated = frame['segment_id'].duplicated()

    if duplicated.any():
        examples = (
            frame.loc[duplicated, 'segment_id']
            .head(10)
            .tolist()
        )
        raise ValueError(
            f'{frame_name} 中存在重复 segment_id：{examples}'
        )

label_ids = set(lab['segment_id'].dropna())
visual_ids = set(pf['segment_id'].dropna())
audio_ids = set(ege['segment_id'].dropna())
common_ids = label_ids & visual_ids & audio_ids

print(
    f"[data] labels={len(lab)} "
    f"visual={len(pf)} "
    f"audio={len(ege)} "
    f"common={len(common_ids)}"
)

print(
    f"[data] labels_without_visual={len(label_ids - visual_ids)} "
    f"labels_without_audio={len(label_ids - audio_ids)}"
)

if not common_ids:
    print('[data] label ID examples:', lab['segment_id'].head().tolist())
    print('[data] visual ID examples:', pf['segment_id'].head().tolist())
    print('[data] audio ID examples:', ege['segment_id'].head().tolist())
    raise ValueError(
        '标签表与特征表没有共同的 segment_id。'
    )

df = (
    lab[['segment_id', 'label', 'mode', 'text']]
    .merge(
        pf,
        on='segment_id',
        how='inner',
        validate='one_to_one',
    )
    .merge(
        ege,
        on='segment_id',
        how='inner',
        validate='one_to_one',
    )
)

print(
    f"[data] merged={len(df)} "
    f"mode_counts={df['mode'].value_counts().to_dict()}"
)

TE = df[
    (df['mode'] == 'test')
    & df['label'].notna()
    & (df['label'] != 0)
].copy()

if MAXN:
    TE = TE.head(MAXN)

print(f'[test] binary samples={len(TE)}')

if TE.empty:
    raise ValueError(
        '测试集为空，请检查 mode、label 和 segment_id 匹配结果。'
    )


# ============================================================
# Raw feature columns + p/r selection
# ============================================================

DROP_FEATS = ['det_rate']

AU_CAND = [
    c for c in pf.columns
    if c != 'segment_id'
    and c not in DROP_FEATS
    and pd.api.types.is_numeric_dtype(pf[c])
]

EG_CAND = [
    c for c in ege.columns
    if c != 'segment_id'
    and c not in DROP_FEATS
    and pd.api.types.is_numeric_dtype(ege[c])
]

# 只用 train 做 p/r 统计，避免 valid/test 泄漏；
# 当前任务是二分类，因此 y 用 label > 0。
FS = df[df['mode'] == 'train'].copy()

if KB_BINARY_ONLY:
    FS = FS[FS['label'] != 0].copy()

y_fs = (FS['label'].astype(float) > 0).astype(int).to_numpy()

FP = {}
FCORR = {}

for c in EG_CAND + AU_CAND:
    x = pd.to_numeric(FS[c], errors='coerce').astype(float).to_numpy()
    m = np.isfinite(x) & np.isfinite(y_fs)

    if m.sum() < 3 or np.nanstd(x[m]) == 0:
        FP[c] = 1.0
        FCORR[c] = 0.0
        continue

    r, p = pearsonr(x[m], y_fs[m])

    if not np.isfinite(r):
        r = 0.0

    if not np.isfinite(p):
        p = 1.0

    FP[c] = float(p)
    FCORR[c] = float(r)


def select(cols):
    """
    p < P_THRESH 准入；
    """
    sel = [
        c for c in cols
        if FP.get(c, 1.0) < P_THRESH
    ]

    return sel


# prompt、补全均值仍使用筛选后的列，以保持提示长度和原有补全输出不变。
AU_SEL = select(AU_CAND)
EG_SEL = select(EG_CAND)

# A/V 最近邻检索单独使用全部候选数值特征，用于检验“不筛选特征”
# 对检索效果的实际影响。这里不改变 prompt、强度分级或被补全的列。
AU_RETRIEVAL = list(AU_CAND)
EG_RETRIEVAL = list(EG_CAND)

print(
    f"[features] visual={len(AU_SEL)}/{len(AU_CAND)} "
    f"acoustic={len(EG_SEL)}/{len(EG_CAND)} "
    f"(p<{P_THRESH}, sorted by |r|)"
)

print(
    f"[retrieval-features] visual={len(AU_RETRIEVAL)} "
    f"acoustic={len(EG_RETRIEVAL)} (all candidate features)"
)


# ============================================================
# Raw feature values to text
# ============================================================

def to_raw(val):
    try:
        v = float(val)

        if pd.isna(v):
            print("over")
            return "missing"

        return f"{v:.2f}"

    except Exception:
        return "missing"


def feat_to_text(row, cols):
    return ', '.join(
        f'{c}={to_raw(row[c])}'
        for c in cols
    )

# ============================================================
# A/V strength grading
# ============================================================

# 最后一步统一做 A/V 强度转化：
# complete_sample 先完成 text/audio/visual 补全；
# 然后 prompt 里凡是出现的 Audio / Visual 数值，全部转成三级强度：
# low / medium / high。
# 阈值只用 train 集计算，避免 valid/test 泄漏。
STRENGTH_LABELS = [
    "low",
    "medium",
    "high",
]


def build_strength_bins(cols):
    bins = {}

    for c in cols:
        vals = pd.to_numeric(FS[c], errors='coerce').astype(float)
        vals = vals[np.isfinite(vals)]

        if len(vals) < 3 or np.nanstd(vals) == 0:
            bins[c] = None
            continue

        # 三级分箱：33.3% / 66.7% 两个分位点。
        qs = np.nanquantile(vals, [1.0 / 3.0, 2.0 / 3.0])

        if not np.all(np.isfinite(qs)) or qs[0] == qs[1]:
            bins[c] = None
        else:
            bins[c] = [float(x) for x in qs]

    return bins


STRENGTH_BINS = build_strength_bins(EG_SEL + AU_SEL)


def to_strength(val, col):
    try:
        v = float(val)

        if not np.isfinite(v):
            return "missing"

        qs = STRENGTH_BINS.get(col, None)

        # 如果某个特征无法稳定分箱，默认给 medium，避免中断流程。
        if qs is None:
            return "medium"

        if v <= qs[0]:
            return "low"

        if v <= qs[1]:
            return "medium"

        return "high"

    except Exception:
        return "missing"


def feat_to_strength_text(row, cols):
    return ', '.join(
        f'{c}={to_strength(row[c], c)}'
        for c in cols
    )


# ============================================================
# Build knowledge bank
# ============================================================

# 补全知识库只用 train，不用 valid / test
KB = df[df['mode'] == 'train'].copy()

if KB_BINARY_ONLY:
    KB = KB[KB['label'] != 0].copy()

KB = KB.reset_index(drop=True)

print(
    f"[kb] samples={len(KB)} "
    f"kb_mode=train "
    f"binary_only={KB_BINARY_ONLY}"
)


# ============================================================
# Text embedding cache
# ============================================================

def sha1_list(xs):
    h = hashlib.sha1()

    for x in xs:
        h.update(str(x).encode('utf-8'))
        h.update(b'\n')

    return h.hexdigest()


def make_text_kb_meta(KB):
    segment_ids = KB['segment_id'].astype(str).tolist()
    texts = KB['text'].fillna('').astype(str).tolist()

    return {
        'model_path': getattr(local_llm, 'MODEL_PATH', 'unknown'),
        'n_samples': len(KB),
        'kb_mode': 'train',
        'kb_binary_only': KB_BINARY_ONLY,
        'text_emb_batch_size': TEXT_EMB_BATCH_SIZE,
        'text_emb_max_length': TEXT_EMB_MAX_LENGTH,
        'segment_hash': sha1_list(segment_ids),
        'text_hash': sha1_list(texts),
    }


def same_meta(a, b):
    keys = [
        'model_path',
        'n_samples',
        'kb_mode',
        'kb_binary_only',
        'text_emb_max_length',
        'segment_hash',
        'text_hash',
    ]

    return all(a.get(k) == b.get(k) for k in keys)


def build_or_load_kb_text_embeddings(KB, force_rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)

    cur_meta = make_text_kb_meta(KB)

    if (
        not force_rebuild
        and os.path.exists(TEXT_EMB_CACHE)
        and os.path.exists(TEXT_META_CACHE)
    ):
        try:
            old_meta = json.load(open(TEXT_META_CACHE, 'r', encoding='utf-8'))

            if same_meta(cur_meta, old_meta):
                data = np.load(TEXT_EMB_CACHE, allow_pickle=False)

                X = data['embeddings'].astype(np.float32)
                segment_ids = data['segment_ids'].astype(str).tolist()

                if X.shape[0] != len(KB):
                    print(
                        f"[text-emb] cache row mismatch: "
                        f"cache={X.shape[0]}, KB={len(KB)}, rebuild"
                    )
                else:
                    print(
                        f"[text-emb] loaded cache: {TEXT_EMB_CACHE}, "
                        f"shape={X.shape}"
                    )
                    return X, segment_ids

            else:
                print("[text-emb] cache meta mismatch, rebuild embeddings")

        except Exception as e:
            print(f"[text-emb] failed to load cache, rebuild: {repr(e)}")

    texts = KB['text'].fillna('').astype(str).tolist()
    segment_ids = KB['segment_id'].astype(str).tolist()

    print(f"[text-emb] building embeddings for {len(texts)} KB texts...")

    X = local_llm.embed(
        texts,
        batch_size=TEXT_EMB_BATCH_SIZE,
        max_length=TEXT_EMB_MAX_LENGTH
    ).astype(np.float32)

    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    np.savez_compressed(
        TEXT_EMB_CACHE,
        embeddings=X,
        segment_ids=np.array(segment_ids)
    )

    json.dump(
        cur_meta,
        open(TEXT_META_CACHE, 'w', encoding='utf-8'),
        ensure_ascii=False,
        indent=2
    )

    print(f"[text-emb] saved -> {TEXT_EMB_CACHE}")
    print(f"[text-emb] saved -> {TEXT_META_CACHE}")

    return X, segment_ids


KB_TEXT_VEC, KB_TEXT_SEGMENT_IDS = build_or_load_kb_text_embeddings(
    KB,
    force_rebuild=FORCE_REBUILD_TEXT_EMB
)

assert KB_TEXT_SEGMENT_IDS == KB['segment_id'].astype(str).tolist(), \
    "KB text embedding order does not match KB rows"


# ============================================================
# A/V feature index for nearest neighbor completion
# ============================================================

# 检索距离的均值/标准差按全部候选特征计算；筛选后的 EG_SEL/AU_SEL
# 仍只负责补全输出和 prompt，二者用途彼此分离。
ALL_NUM_COLS = EG_RETRIEVAL + AU_RETRIEVAL

MU = {}
SD = {}

for c in ALL_NUM_COLS:
    vals = pd.to_numeric(KB[c], errors='coerce').astype(float)

    mu = vals.mean()
    sd = vals.std()

    if not np.isfinite(mu):
        mu = 0.0

    if not np.isfinite(sd) or sd == 0:
        sd = 1.0

    MU[c] = mu
    SD[c] = sd


def available_index_cols(missing):
    """
    当前 condition 可见的全部 A/V 候选特征作为检索索引。
    text 不在这里处理，text 检索单独走 embedding。
    """
    cols = []

    if 'audio' not in missing:
        cols += EG_RETRIEVAL

    if 'visual' not in missing:
        cols += AU_RETRIEVAL

    return cols


def row_vec(row, cols):
    vec = []

    for c in cols:
        try:
            v = float(row[c])
        except Exception:
            v = np.nan

        if not np.isfinite(v):
            vec.append(np.nan)
        else:
            vec.append((v - MU[c]) / SD[c])

    return np.array(vec, dtype=float)


_KB_VEC_CACHE = {}


def get_kb_matrix(cols):
    key = tuple(cols)

    if key not in _KB_VEC_CACHE:
        _KB_VEC_CACHE[key] = np.vstack([
            row_vec(r, cols)
            for _, r in KB.iterrows()
        ])

    return _KB_VEC_CACHE[key]


def retrieve_neighbors_by_av(row, cols, topk=3):
    """
    用当前可见 A/V 特征找最近邻。
    距离为 nan-aware standardized Euclidean distance。
    """
    if not cols:
        return [], []

    q = row_vec(row, cols)
    X = get_kb_matrix(cols)

    q_mask = np.isfinite(q)
    X_mask = np.isfinite(X)

    valid = X_mask & q_mask[None, :]
    cnt = valid.sum(axis=1)

    ok = cnt >= 1

    if 'segment_id' in row:
        ok &= (KB['segment_id'].values != row['segment_id'])

    diff = X - q[None, :]
    diff[~valid] = 0.0

    dist = np.sqrt((diff ** 2).sum(axis=1) / np.maximum(cnt, 1))
    dist[~ok] = np.inf

    idx = np.argsort(dist)
    idx = [int(i) for i in idx if np.isfinite(dist[i])]

    if not idx:
        return [], []

    idx = idx[:topk]

    neighbors = [KB.iloc[i] for i in idx]
    dists = [float(dist[i]) for i in idx]

    return neighbors, dists


# ============================================================
# Text nearest neighbor completion
# ============================================================

def has_valid_text(row):
    if 'text' not in row:
        return False

    t = row.get('text', None)

    if pd.isna(t):
        return False

    t = str(t).strip()

    return len(t) > 0 and t.lower() != 'nan'


def retrieve_neighbors_by_text(row, topk=3):
    """
    当前样本有 text 时，用 Qwen hidden-state embedding 找语义最近邻。
    """
    if not has_valid_text(row):
        return [], []

    q_text = str(row['text'])

    q_vec = local_llm.embed(
        q_text,
        batch_size=1,
        max_length=TEXT_EMB_MAX_LENGTH
    ).astype(np.float32)

    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

    sims = KB_TEXT_VEC @ q_vec

    ok = np.ones(len(KB), dtype=bool)

    if 'segment_id' in row:
        ok &= (KB['segment_id'].values != row['segment_id'])

    sims = sims.copy()
    sims[~ok] = -np.inf

    idx = np.argsort(-sims)
    idx = [int(i) for i in idx if np.isfinite(sims[i])]

    if not idx:
        return [], []

    idx = idx[:topk]

    neighbors = [KB.iloc[i] for i in idx]

    # 为了和原来的 distances 字段兼容，用 1 - cosine_similarity 当距离
    dists = [float(1.0 - sims[i]) for i in idx]

    return neighbors, dists


# ============================================================
# Imputation
# ============================================================

def mean_impute_numeric(new_row, neighbors, cols):
    """
    用 top-k 邻居均值补数值模态。
    """
    for c in cols:
        vals = []

        for nb in neighbors:
            try:
                v = float(nb[c])
            except Exception:
                continue

            if np.isfinite(v):
                vals.append(v)

        if vals:
            new_row[c] = float(np.mean(vals))


def complete_sample(row, cond, missing):
    """
    补全逻辑：

    如果当前样本有 text:
        优先用 text embedding 检索 KB。
        用语义近邻补缺失的 audio / visual。

    如果当前样本没有 text:
        退回 A/V KNN。
        用 A/V 近邻补缺失的 text / audio / visual。

    condition 对应关系：
        T:
            text -> 补 audio + visual，走 text 检索。

        A:
            audio -> 补 text + visual，走 A/V 检索。

        V:
            visual -> 补 text + audio，走 A/V 检索。

        TA:
            text + audio -> 补 visual，优先走 text 检索。

        TV:
            text + visual -> 补 audio，优先走 text 检索。

        AV:
            audio + visual -> 补 text，走 A/V 检索。

        TAV:
            不补。
    """
    new_row = row.copy()

    meta = {
        'used_completion': False,
        'retrieval_mode': None,

        'neighbor_segment_ids': [],
        'distances': [],

        'imputed_text': False,
        'imputed_audio': False,
        'imputed_visual': False,

        'text_source': 'original' if 'text' not in missing else 'missing',
        'audio_source': 'original' if 'audio' not in missing else 'missing',
        'visual_source': 'original' if 'visual' not in missing else 'missing',

        'reason': None,
    }

    if cond not in COMPLETION_CONDS:
        meta['reason'] = f'{cond} does not use completion'
        return new_row, meta

    neighbors = []
    dists = []
    retrieval_mode = None

    # 1. 有 text 时，优先用 text embedding 检索
    if 'text' not in missing and has_valid_text(row):
        neighbors, dists = retrieve_neighbors_by_text(row, topk=TOPK)

        if neighbors:
            retrieval_mode = 'text'

    # 2. 没有 text 或 text 检索失败，则退回 A/V KNN
    if not neighbors:
        index_cols = available_index_cols(missing)

        if not index_cols:
            meta['reason'] = 'no available text or A/V index features'
            return new_row, meta

        neighbors, dists = retrieve_neighbors_by_av(
            row,
            index_cols,
            topk=TOPK
        )

        if neighbors:
            retrieval_mode = 'audio_visual'

    if not neighbors:
        meta['reason'] = 'no valid neighbor'
        return new_row, meta

    # text 缺失时，用最近邻 text 补
    # text 缺失时，用 top-k 邻居 text 拼接补
    if 'text' in missing:
        new_row['text'] = ' [SEP] '.join(
            str(nb['text'])
            for nb in neighbors[:TOPK]
        )
        meta['imputed_text'] = True
        meta['text_source'] = f'imputed_by_{retrieval_mode}'

    # audio 缺失时，用 top-k 均值补
    if 'audio' in missing:
        mean_impute_numeric(new_row, neighbors, EG_SEL)
        meta['imputed_audio'] = True
        meta['audio_source'] = f'imputed_by_{retrieval_mode}'

    # visual 缺失时，用 top-k 均值补
    if 'visual' in missing:
        mean_impute_numeric(new_row, neighbors, AU_SEL)
        meta['imputed_visual'] = True
        meta['visual_source'] = f'imputed_by_{retrieval_mode}'

    meta['used_completion'] = (
        meta['imputed_text']
        or meta['imputed_audio']
        or meta['imputed_visual']
    )

    meta['retrieval_mode'] = retrieval_mode
    meta['neighbor_segment_ids'] = [str(nb['segment_id']) for nb in neighbors]
    meta['distances'] = [round(float(d), 6) for d in dists]
    meta['reason'] = f'completed by {retrieval_mode} nearest neighbors'

    return new_row, meta


# ============================================================
# Prompt
# ============================================================

def completed_prompt(row, cond, missing):
    """
    补全后仍然按原始格式输入：
        Text:
        Audio features:
        Visual features:

    注意：
        不建议把 imputed_by_text / imputed_by_audio_visual 写进 prompt。
        这些只放进 log，避免影响大模型判断。
    """
    completed_row, meta = complete_sample(row, cond, missing)

    text_available = ('text' not in missing) or meta['imputed_text']
    audio_available = ('audio' not in missing) or meta['imputed_audio']
    visual_available = ('visual' not in missing) or meta['imputed_visual']

    text = str(completed_row['text']) if text_available else None

    # 最后一步再做强度转化：
    # complete_sample 已经完成 text/audio/visual 补全；
    # 这里对最终 prompt 里所有可用的 A/V 数值统一转成三级强度。
    # 因此只影响送入 LLM 的文本表示，不影响前面的检索、均值补全和日志统计。
    audio = feat_to_strength_text(completed_row, EG_SEL) if audio_available else None
    visual = feat_to_strength_text(completed_row, AU_SEL) if visual_available else None

    parts = []

    if text is not None:
        parts.append(f"Text: {text}")

    if audio is not None:
        parts.append(f"Audio features: {audio}")

    if visual is not None:
        parts.append(f"Visual features: {visual}")

    body = ', '.join(parts)

    return (
        f"Classify the sentiment polarity of this utterance. {body}, "
        f"Answer with exactly one word: Positive or Negative. Without other text."
    ), meta


# ============================================================
# Parse output
# ============================================================

def parse_polarity(t):
    """
    1 = positive
    0 = negative

    如果没有解析到类别词，或者 positive/negative 同时出现，
    返回 None。后续评估时 None 会被计为错误，而不是跳过。
    """
    s = str(t)
    tl = s.lower()

    pos = ('positive' in tl) or ('正面' in s) or ('积极' in s)
    neg = ('negative' in tl) or ('负面' in s) or ('消极' in s)

    if pos and not neg:
        return 1

    if neg and not pos:
        return 0

    # 没有类别词，或 positive/negative 同时出现，都视为解析失败
    return None


# ============================================================
# Evaluation
# ============================================================

print(
    f"{'cond':<6}{'Acc2':>8}{'F1':>8}{'valid/tot':>11} "
    f"[raw feature values, text-first completion, A/V fallback, invalid-as-wrong]"
)

rows = []
completion_logs = []

for cond, missing in CONDITIONS.items():
    yt, yp = [], []
    ninv = 0

    n_completed = 0
    n_text = 0
    n_audio = 0
    n_visual = 0

    n_retrieval_text = 0
    n_retrieval_av = 0

    for _, r in TE.iterrows():
        gold = 1 if r['label'] > 0 else 0

        try:
            prompt, meta = completed_prompt(r, cond, missing)

            if meta['used_completion']:
                n_completed += 1

            if meta['imputed_text']:
                n_text += 1

            if meta['imputed_audio']:
                n_audio += 1

            if meta['imputed_visual']:
                n_visual += 1

            if meta['retrieval_mode'] == 'text':
                n_retrieval_text += 1

            if meta['retrieval_mode'] == 'audio_visual':
                n_retrieval_av += 1

            completion_logs.append({
                'condition': cond,
                'segment_id': r['segment_id'],
                'label': float(r['label']),

                'used_completion': meta['used_completion'],
                'retrieval_mode': meta['retrieval_mode'],

                'neighbor_segment_ids': meta['neighbor_segment_ids'],
                'distances': meta['distances'],

                'imputed_text': meta['imputed_text'],
                'imputed_audio': meta['imputed_audio'],
                'imputed_visual': meta['imputed_visual'],

                'text_source': meta['text_source'],
                'audio_source': meta['audio_source'],
                'visual_source': meta['visual_source'],

                'reason': meta['reason'],
            })

            out = local_llm.generate(prompt, max_new_tokens=20)
            pred = parse_polarity(out)

        except Exception as e:
            pred = None

            completion_logs.append({
                'condition': cond,
                'segment_id': r.get('segment_id', None),
                'label': float(r['label']) if 'label' in r else None,

                'used_completion': False,
                'retrieval_mode': None,

                'neighbor_segment_ids': [],
                'distances': [],

                'imputed_text': False,
                'imputed_audio': False,
                'imputed_visual': False,

                'text_source': 'unknown',
                'audio_source': 'unknown',
                'visual_source': 'unknown',

                'reason': f'exception: {repr(e)}',
            })

        # gold 一定进入分母
        yt.append(gold)

        if pred is None:
            ninv += 1

            # invalid 输出不是模型真的预测了反类；
            # 这里人为填入 gold 的相反标签，只是为了让 sklearn 的 Acc/F1
            # 把该样本计为错误，而不是跳过。
            yp.append(1 - gold)
            continue

        yp.append(pred)

    if yt:
        a = accuracy_score(yt, yp)
        f = f1_score(yt, yp, average='weighted', zero_division=0)

        print(
            f"{cond:<7}{a:>8.3f}{f:>8.3f}"
            f"{f'{len(yt)}/{len(TE)}':>11} "
            f"(invalid_as_wrong={ninv}, completed={n_completed}, "
            f"text={n_text}, audio={n_audio}, visual={n_visual}, "
            f"retrieval_text={n_retrieval_text}, "
            f"retrieval_av={n_retrieval_av})"
        )

        rows.append({
            'condition': cond,
            'acc2': round(a, 3),
            'f1': round(f, 3),
            'n_total': len(yt),
            'invalid_as_wrong': ninv,

            'completed': n_completed,
            'imputed_text': n_text,
            'imputed_audio': n_audio,
            'imputed_visual': n_visual,

            'retrieval_text': n_retrieval_text,
            'retrieval_audio_visual': n_retrieval_av,
        })


# ============================================================
# Save
# ============================================================

RESULT_PATH = 'baseline_raw_all_text_first_completed_results.json'
LOG_PATH = 'baseline_raw_all_text_first_completion_logs.json'

json.dump(
    rows,
    open(RESULT_PATH, 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2
)

json.dump(
    completion_logs,
    open(LOG_PATH, 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2
)

print(f"\nsaved -> {RESULT_PATH}")
print(f"saved -> {LOG_PATH}")
