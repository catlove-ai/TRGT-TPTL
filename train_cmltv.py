import logging
import os.path

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.base_model import BaseModel
from modeling.cmltv_model import CMLTV_Model
from utils.common import get_root_path
from utils.datasets import BaseDataset, base_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def train(jf_game_id, train_start_date, train_end_date, val_start_date, val_end_date, save_path, features, device='cuda:0', max_row=None, minus_label_col=None):
    # args
    epochs = 20
    batch_size = 1024

    lrd = {
        'wm': 0.002,
        'wxsh': 0.002,
        'mdnf': 0.002,
        'cmelive': 0.001
    }

    lr = lrd[jf_game_id]
    lr_scheduler_patience = 0
    early_stop_patience = 2

    # log feature info
    feature_number = {
        'dense': len(features['dense']),
        'sparse': len(features['sparse']),
        'seq': len(features['seq'])
    }
    logging.info(f'feature number:{feature_number}')

    # model
    model = CMLTV_Model(features=features, device=device)

    # optim
    optimizer = optim.Adam(model.parameters(), lr)
    # scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, min_lr=1e-6)
    scheduler = None
    early_stopping = EarlyStopping(patience=early_stop_patience)

    # data
    train_dataset = BaseDataset(jf_game_id, train_start_date, train_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=base_collate)
    val_dataset = BaseDataset(jf_game_id, val_start_date, val_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=base_collate)

    # train
    trainer = Trainer(model, optimizer, scheduler, early_stopping)
    trainer.train(save_path, epochs, train_loader, val_loader)


if __name__ == '__main__':
    set_seed(42)
    # for jf_game_id in ['wxsh']:
    #     for method in ['DNN', 'MoE', 'DCN']:
    #         save_path = os.path.join(get_root_path(), 'checkpoints/wxsh', f'{jf_game_id}_{method}_active_th_seed_42')
    #         train(jf_game_id, 805, 825, 826, 829,
    #               save_path=save_path, features=ACTIVE_FEATURES, minus_label_col=None, model=method, max_row=None)
    for jf_game_id in ['cmelive']:
        save_path = os.path.join(get_root_path(), f'checkpoints/{jf_game_id}', f'{jf_game_id}_cmltv_6h_tc_seed_42_test')
        train(jf_game_id, 805, 818, 819, 820,
              save_path=save_path, features=ALL_FEATURES, minus_label_col='ltv_6h', max_row=None)
    # for jf_game_id in ['mdnf']:
    #     for method in ['DCN']:
    #         save_path = os.path.join(get_root_path(), 'checkpoints/train_model', f'{jf_game_id}_{method}_active_th_seed_42')
    #         train(jf_game_id, 812, 825, 826, 827,
    #             save_path=save_path, features=ACTIVE_FEATURES, minus_label_col=None, model=method, max_row=None)
