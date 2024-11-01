import torch


class CategoricalFeature:
    def __init__(self, idx, name, vocab_size, version=1):
        self.idx = idx
        self.name = name
        self.vocab_size = vocab_size
        self.version = version


class ConfigFeature:
    def __init__(self, idx, name, dtype):
        self.idx = idx
        self.name = name
        self.dtype = dtype


class LabelFeature:
    def __init__(self, idx, name, dtype=torch.float32):
        self.idx = idx
        self.name = name
        self.dtype = dtype


class SeqFeature:
    def __init__(self, name, max_len, vocab_size, dtype=torch.float32, emb_size=4, version=1):
        self.name = name
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.emb_size = emb_size
        self.version = version


# 7
LABELS = [
    LabelFeature(0, 'ltv_3h', torch.float32),
    LabelFeature(1, 'ltv_6h', torch.float32),
    LabelFeature(2, 'ltv_12h', torch.float32),
    LabelFeature(3, 'ltv_24h', torch.float32),
    LabelFeature(4, 'ltv_36h', torch.float32),
    LabelFeature(5, 'ltv3', torch.float32),
    LabelFeature(6, 'ltv7', torch.float32),
    LabelFeature(7, 'retain2', torch.int64)  # 留存
]

# 5
LOGIN_CONFIG = [  # 不参与训练
    ConfigFeature(idx=0, name='vopenid', dtype=str),
    ConfigFeature(idx=1, name='vgameappid', dtype=str),
    ConfigFeature(idx=2, name='o2_game_id', dtype=str),
    ConfigFeature(idx=3, name='media_type', dtype=str),
    ConfigFeature(idx=4, name='media_id', dtype=str),
]

# 3
LOGIN_CONFIG_HASH = [
    CategoricalFeature(idx=0, name='o2_game_id_hash', vocab_size=1000),
    CategoricalFeature(idx=0, name='media_type_hash', vocab_size=100),
    CategoricalFeature(idx=0, name='media_id_hash', vocab_size=1000),
]

# 4
LOGIN_DENSE = [
    'screenwidth',
    'screenheight',
    'density',
    'memory',
]

# 8
LOGIN_SPARSE = [
    CategoricalFeature(idx=0, name='user_type', vocab_size=10),
    CategoricalFeature(idx=0, name='clientversion', vocab_size=100),
    CategoricalFeature(idx=0, name='systemsoftware', vocab_size=100),
    CategoricalFeature(idx=0, name='systemhardware', vocab_size=1000),
    CategoricalFeature(idx=0, name='telecomoper', vocab_size=10),
    CategoricalFeature(idx=0, name='network', vocab_size=10),
    CategoricalFeature(idx=0, name='loginchannel', vocab_size=1000),
    CategoricalFeature(idx=0, name='gamesvrid', vocab_size=1000),
    CategoricalFeature(idx=0, name='platid', vocab_size=10),
]

# 14
GAME_BEHAVIOR_DENSE = [
    'login_cnt',
    'onlinetime_sum',
    'onlinetime_ave',
    'moneyflow_add_cnt',
    'moneyflow_sub_cnt',
    'moneyflow_add_value',
    'moneyflow_sub_value',
    'moneyflow_add_type',
    'moneyflow_sub_type',
    'taskflow_count',
    'taskflow_type',
    'level',
    'friendnum',
    'vip_level'
]

# 17
DAPAN_SPARSE = [
    CategoricalFeature(10, "province_index", 100, version=1),
    CategoricalFeature(11, "city_level_index", 20, version=1),
    CategoricalFeature(12, "grade_index", 10, version=1),
    CategoricalFeature(13, "constellation_index", 100, 4),
    CategoricalFeature(14, "rural_crowd_index", 10, 4),
    CategoricalFeature(15, "high_profile_app_crowd_index", 10, 4),
    CategoricalFeature(16, "people_with_cars_index", 10, 4),
    CategoricalFeature(17, "people_with_house_index", 10, version=1),
    CategoricalFeature(18, "lifestatus_v2_index", 10, 4),
    CategoricalFeature(19, "gender_v5_index", 10, version=1),
    CategoricalFeature(20, "age_range", 10, 4),
    CategoricalFeature(21, "age_v5", 120, version=1),
    CategoricalFeature(22, "animeflag", 10, version=1),
    CategoricalFeature(23, "sum_payment_year_bucket", 10, version=1),
    CategoricalFeature(24, "sum_payment_30days_bucket", 10, version=1),
    CategoricalFeature(25, "sum_pay_times_year_bucket", 10, version=1),
    CategoricalFeature(26, "sum_pay_times_30days_bucket", 10, version=1)
]

# 220
DAPAN_DENSE = [
    "active_game_count",
    "active_count",
    "active_count_quantile",
    "active_smoba",
    "active_jdqssy",
    "active_jk",
    "active_hlddzh5",
    "active_lgame",
    "active_poker",
    "active_uamobile_live",
    "active_chinesechess",
    "active_af",
    "active_kihan",
    "active_cmelive",
    "active_clashofclans",
    "active_qsm",
    "active_mahjong",
    "active_hlmj3dh5",
    "active_wepang",
    "active_bs",
    "active_clashroyale",
    "active_love",
    "active_ffm",
    "active_wefeng",
    "active_herokill",
    "active_nbamg",
    "active_qqx5",
    "active_wepop",
    "active_ttzq",
    "active_wxsh",
    "active_hlup",
    "active_mg",
    "active_tlbb",
    "active_xgame2",
    "active_wefly",
    "active_hjol",
    "active_qjnn",
    "active_vc",
    "active_shootgame",
    "active_wslg",
    "active_wuxia_mobile",
    "active_ttmj",
    "active_swy",
    "active_kr",
    "active_tlbb2",
    "active_egame",
    "active_byjg",
    "active_wmsj",
    "active_qhmy",
    "active_kof",
    "active_smbbxsj",
    "active_hxgame",
    "active_newxgame",
    "active_wefire",
    "active_jxqy",
    "active_wpzs",
    "active_mirmobile",
    "active_ffom",
    "active_ssk",
    "active_xssh",
    "active_vxd",
    "active_sgqyz",
    "active_lzjd",
    "active_gwgo",
    "active_dpcqm",
    "active_lol",
    "active_cf",
    "active_qqkart",
    "active_dnf",
    "active_x5",
    "active_tgame",
    "active_fo4",
    "active_nba2kol2",
    "active_cfhd",
    "active_newroco",
    "active_sg",
    "active_nba",
    "active_qsgame",
    "active_x52",
    "active_bns",
    "active_sopcn",
    "active_tps",
    "active_ffoqq",
    "active_hy",
    "active_yl",
    "active_wgame",
    "active_hjdt",
    "pay_game_count",
    "pay_count",
    "pay_sum",
    "pay_sum_quantile",
    "payment_smoba",
    "payment_jdqssy",
    "payment_jk",
    "payment_hlddzh5",
    "payment_lgame",
    "payment_poker",
    "payment_uamobile_live",
    "payment_chinesechess",
    "payment_af",
    "payment_kihan",
    "payment_cmelive",
    "payment_clashofclans",
    "payment_qsm",
    "payment_mahjong",
    "payment_hlmj3dh5",
    "payment_wepang",
    "payment_bs",
    "payment_clashroyale",
    "payment_love",
    "payment_ffm",
    "payment_wefeng",
    "payment_herokill",
    "payment_nbamg",
    "payment_qqx5",
    "payment_wepop",
    "payment_ttzq",
    "payment_wxsh",
    "payment_hlup",
    "payment_mg",
    "payment_tlbb",
    "payment_xgame2",
    "payment_wefly",
    "payment_hjol",
    "payment_qjnn",
    "payment_vc",
    "payment_shootgame",
    "payment_wslg",
    "payment_wuxia_mobile",
    "payment_ttmj",
    "payment_swy",
    "payment_kr",
    "payment_tlbb2",
    "payment_egame",
    "payment_byjg",
    "payment_wmsj",
    "payment_qhmy",
    "payment_kof",
    "payment_smbbxsj",
    "payment_hxgame",
    "payment_newxgame",
    "payment_wefire",
    "payment_jxqy",
    "payment_wpzs",
    "payment_mirmobile",
    "payment_ffom",
    "payment_ssk",
    "payment_xssh",
    "payment_vxd",
    "payment_sgqyz",
    "payment_lzjd",
    "payment_gwgo",
    "payment_dpcqm",
    "payment_lol",
    "payment_cf",
    "payment_qqkart",
    "payment_dnf",
    "payment_x5",
    "payment_tgame",
    "payment_fo4",
    "payment_nba2kol2",
    "payment_cfhd",
    "payment_newroco",
    "payment_sg",
    "payment_nba",
    "payment_qsgame",
    "payment_x52",
    "payment_bns",
    "payment_sopcn",
    "payment_tps",
    "payment_ffoqq",
    "payment_hy",
    "payment_yl",
    "payment_wgame",
    "payment_hjdt",
    "2_8_1",  # 正常 0.2gini
    "2_8_10",
    "2_8_11",
    "2_8_12",
    "2_8_13",
    "2_8_14",
    "2_8_15",
    "2_8_16",
    "2_8_17",
    "2_8_18",
    "2_8_19",
    "2_8_2",
    "2_8_20",
    "2_8_21",
    "2_8_3",
    "2_8_4",
    "2_8_5",
    "2_8_6",
    "2_8_7",
    "2_8_8",
    "2_8_9",
    "mean_level",  # 不稳定
    "max_level",
    "max_register_days",
    "min_register_days",
    "max_last_active_days",
    "min_last_active_days",
    "mean_active_days_ratio",
    "max_active_days_ratio",
    "mean_active_weeks_ratio",
    "max_active_weeks_ratio",
    "mean_active_months_ratio",
    "max_active_months_ratio",
    "pay_game_num_cumulative",
    "sum_payment_cumulative",
    "sum_pay_times_cumulative",
    "pay_game_nums_year",
    "sum_payment_year",
    "sum_pay_times_year",
    "pay_game_nums_180days",
    "sum_payment_180days",
    "sum_pay_times_180days",
    "pay_game_nums_30days",
    "sum_payment_30days",
    "sum_pay_times_30days"
]

# 5
DAPAN_SEQ = [
    SeqFeature("register_game_seq", 20, 1000, torch.int64, version=2),
    SeqFeature("active_game_seq", 20, 1000, torch.int64, version=2),
    SeqFeature("onlinetime_seq", 20, 10, torch.float32, version=2),  # 需自行特征工程
    SeqFeature("pay_game_seq", 20, 1000, torch.int64, version=2),
    SeqFeature("payment_seq", 20, 10, torch.float32, version=2),  # 需自行特征工程
]

# 10
TLOG_SPARSE = [
    CategoricalFeature(0, "cat_spe_feat1", 1000, version=1),
    CategoricalFeature(1, "cat_spe_feat2", 1000, version=1),
    CategoricalFeature(2, "cat_spe_feat3", 1000, version=1),
    CategoricalFeature(3, "cat_spe_feat4", 1000, version=1),
    CategoricalFeature(4, "cat_spe_feat5", 1000, version=1),
    CategoricalFeature(5, "cat_spe_feat6", 1000, version=1),
    CategoricalFeature(6, "cat_spe_feat7", 1000, version=1),
    CategoricalFeature(7, "cat_spe_feat8", 1000, version=1),
    CategoricalFeature(8, "cat_spe_feat9", 1000, version=1),
    CategoricalFeature(9, "cat_spe_feat10", 1000, version=1)
]

TLOG_SPARSE_ROLE_NUM = [
    CategoricalFeature(0, "cat_spe_feat1", 1000, version=1),
    CategoricalFeature(1, "cat_spe_feat2", 1000, version=1),
    CategoricalFeature(2, "cat_spe_feat3", 1000, version=1),
    CategoricalFeature(3, "cat_spe_feat4", 1000, version=1),
    CategoricalFeature(4, "cat_spe_feat5", 1000, version=1),
    CategoricalFeature(5, "cat_spe_feat6", 1000, version=1),
    CategoricalFeature(6, "cat_spe_feat7", 1000, version=1),
    CategoricalFeature(7, "cat_spe_feat8", 1000, version=1),
    CategoricalFeature(8, "cat_spe_feat9", 1000, version=1),
    CategoricalFeature(9, "cat_spe_feat10", 1000, version=1),
    CategoricalFeature(10, "unique_vroleid_count", 100, version=1)
]
# 30
TLOG_DENSE_30 = [
    "dense_spe_feat1",
    "dense_spe_feat2",
    "dense_spe_feat3",
    "dense_spe_feat4",
    "dense_spe_feat5",
    "dense_spe_feat6",
    "dense_spe_feat7",
    "dense_spe_feat8",
    "dense_spe_feat9",
    "dense_spe_feat10",
    "dense_spe_feat11",
    "dense_spe_feat12",
    "dense_spe_feat13",
    "dense_spe_feat14",
    "dense_spe_feat15",
    "dense_spe_feat16",
    "dense_spe_feat17",
    "dense_spe_feat18",
    "dense_spe_feat19",
    "dense_spe_feat20",
    "dense_spe_feat21",
    "dense_spe_feat22",
    "dense_spe_feat23",
    "dense_spe_feat24",
    "dense_spe_feat25",
    "dense_spe_feat26",
    "dense_spe_feat27",
    "dense_spe_feat28",
    "dense_spe_feat29",
    "dense_spe_feat30",
]

# 50
TLOG_DENSE_50 = [
    "dense_spe_feat1",
    "dense_spe_feat2",
    "dense_spe_feat3",
    "dense_spe_feat4",
    "dense_spe_feat5",
    "dense_spe_feat6",
    "dense_spe_feat7",
    "dense_spe_feat8",
    "dense_spe_feat9",
    "dense_spe_feat10",
    "dense_spe_feat11",
    "dense_spe_feat12",
    "dense_spe_feat13",
    "dense_spe_feat14",
    "dense_spe_feat15",
    "dense_spe_feat16",
    "dense_spe_feat17",
    "dense_spe_feat18",
    "dense_spe_feat19",
    "dense_spe_feat20",
    "dense_spe_feat21",
    "dense_spe_feat22",
    "dense_spe_feat23",
    "dense_spe_feat24",
    "dense_spe_feat25",
    "dense_spe_feat26",
    "dense_spe_feat27",
    "dense_spe_feat28",
    "dense_spe_feat29",
    "dense_spe_feat30",
    "dense_spe_feat31",
    "dense_spe_feat32",
    "dense_spe_feat33",
    "dense_spe_feat34",
    "dense_spe_feat35",
    "dense_spe_feat36",
    "dense_spe_feat37",
    "dense_spe_feat38",
    "dense_spe_feat39",
    "dense_spe_feat40",
    "dense_spe_feat41",
    "dense_spe_feat42",
    "dense_spe_feat43",
    "dense_spe_feat44",
    "dense_spe_feat45",
    "dense_spe_feat46",
    "dense_spe_feat47",
    "dense_spe_feat48",
    "dense_spe_feat49",
    "dense_spe_feat50"
]


# 仅大盘特征 + config hash
DAPAN_FEATURES = {
    'config': LOGIN_CONFIG,
    'dense': DAPAN_DENSE,  # 220
    'sparse': DAPAN_SPARSE,  # 3+17=20
    'seq': DAPAN_SEQ,  # 5
    'labels': LABELS
}

# 激活时特征
ACTIVE_FEATURES = {
    'config': LOGIN_CONFIG,
    'dense': LOGIN_DENSE + DAPAN_DENSE,  # 4+220=224
    'sparse': LOGIN_SPARSE + DAPAN_SPARSE +LOGIN_CONFIG_HASH,  # 3+8+17=28
    'seq': DAPAN_SEQ,  # 5
    'labels': LABELS
}

# 完整特征
ALL_FEATURES = {
    'config': LOGIN_CONFIG,
    'dense': LOGIN_DENSE + DAPAN_DENSE + GAME_BEHAVIOR_DENSE + TLOG_DENSE_30,  # 4+220+14+50=288
    'sparse': LOGIN_SPARSE + DAPAN_SPARSE + TLOG_SPARSE +LOGIN_CONFIG_HASH,  # 3+8+17+10=38
    'seq': DAPAN_SEQ,  # 5
    'labels': LABELS
}

Privileged_FEATURES = {
    'dense': GAME_BEHAVIOR_DENSE + TLOG_DENSE_30,
    'sparse': TLOG_SPARSE
}


def get_feature_names(features: dict | list):
    if isinstance(features, dict):
        feature_names = []
        for feature in features['dense']:
            feature_names.append(feature)
        for feature_type in ['sparse', 'seq', 'labels']:
            for feature in features[feature_type]:
                feature_names.append(feature.name)
        return feature_names
    else:
        return [(lambda f: f if isinstance(f, str) else f.name)(f) for f in features]


if __name__ == '__main__':
    print(get_feature_names(DAPAN_SEQ))
