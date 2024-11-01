from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.base_model import BaseModel
from modeling.trtg_model import TRTGModel
from utils.common import get_root_path
from utils.datasets import BaseDataset, base_collate, multi_label_collate, MultiLabelDataset


if __name__ == '__main__':
    jf_game_id = 'mdnf'
    val_start_date = 819
    val_end_date = 820
    features = ACTIVE_FEATURES
    max_row = 100
    minus_label_col = None
    batch_size = 4
    device = 'cuda:0'

    val_dataset = MultiLabelDataset(jf_game_id, val_start_date, val_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_label_collate)

    model = TRTGModel(features=features, device=device, cross_net='MoE')


    batch = next(iter(val_loader))

    out = model(batch)