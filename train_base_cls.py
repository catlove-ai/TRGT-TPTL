import logging
import os.path

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.base_cls_model import Base_CLS_Model
from modeling.trgt_cls_model import TRTG_CLS_Model
from modeling.trtg_model import TRTGModel
from utils.common import get_root_path
from utils.datasets import MultiLabelDataset, multi_label_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def train(jf_game_id, train_start_date, train_end_date, val_start_date, val_end_date, save_path, method, features, w_t, device='cuda:0', max_row=None, minus_label_col=None):
    # args
    epochs = 10
    batch_size = 1024
    lr = 0.005
    lr_scheduler_patience = 100
    early_stop_patience = 2

    # log feature info
    feature_number = {
        'dense': len(features['dense']),
        'sparse': len(features['sparse']),
        'seq': len(features['seq'])
    }
    logging.info(f'feature number:{feature_number}')

    # model
    cls_path = None
    if jf_game_id=='mdnf':
        cls_path = os.path.join(get_root_path(), rf'checkpoints/pretrain_model/{jf_game_id}_pretrain_300w_th_seed_42/20.pth')
    elif jf_game_id=='dhp':
        cls_path = os.path.join(get_root_path(), rf'checkpoints/pretrain_model/{jf_game_id}_pretrain_810~825_th_seed_42/20.pth')
    elif jf_game_id=='wxsh':
        cls_path = os.path.join(get_root_path(), rf'checkpoints/pretrain_model/{jf_game_id}_pretrain_701~825_th_seed_42/20.pth')
    elif jf_game_id=='wm':
        cls_path = os.path.join(get_root_path(), rf'checkpoints/pretrain_model/{jf_game_id}_pretrain_723~914_th_seed_42/20.pth')
    model = Base_CLS_Model(pretrained_weights_path=cls_path, features=features, cross_net=method,  w_t=w_t, device=device)

    # optim
    optimizer = optim.Adam(model.parameters(), lr)
    # optimizer = optim.Adam([{'params': model.pretrained_model.parameters(), 'lr': 0.0001}], lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, min_lr=1e-6)
    scheduler = None
    early_stopping = EarlyStopping(patience=early_stop_patience)

    # data
    train_dataset = MultiLabelDataset(jf_game_id, train_start_date, train_end_date, features=ALL_FEATURES, max_rows=max_row, minus_label_col=minus_label_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_label_collate)
    val_dataset = MultiLabelDataset(jf_game_id, val_start_date, val_end_date, features=ALL_FEATURES, max_rows=max_row, minus_label_col=minus_label_col)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_label_collate)

    # train
    trainer = Trainer(model, optimizer, scheduler, early_stopping)
    trainer.train(save_path, epochs, train_loader, val_loader)


if __name__ == '__main__':
    set_seed(42)
    jf_game_id = 'wm'
    wt = 0.5
    for method in ['DNN','MoE','DCN']:
        save_path = os.path.join(get_root_path(), f'checkpoints/{jf_game_id}', f'{jf_game_id}_{method}_cls_wt{wt}_active_th_seed_42_with_teacher(20)_new')
        train(jf_game_id, 805, 825, 826, 829,
            save_path=save_path, features=ACTIVE_FEATURES, minus_label_col=None, w_t=wt, method=method, max_row=None)

