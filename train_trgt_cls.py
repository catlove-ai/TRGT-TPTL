import logging
import os.path

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.trgt_cls_model import TRTG_CLS_Model
from modeling.trtg_model import TRTGModel
from utils.common import get_root_path
from utils.datasets import MultiLabelDataset, multi_label_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def train(jf_game_id, train_start_date, train_end_date, val_start_date, val_end_date, save_path, features, w_t, device='cuda:0', max_row=None, minus_label_col=None):
    # args
    epochs = 10
    batch_size = 1024
    lr = 0.002
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
    model = TRTG_CLS_Model(pretrained_weights_path=os.path.join(get_root_path(), rf'checkpoints/pretrain_model/{jf_game_id}_pretrain_701~825_th_seed_42/20.pth'), trgt_layers=1, w_t=w_t, aux_loss_weight=0.5, device=device)

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
    jf_game_id = 'wxsh'
    for pretrain_data in [500000, 1000000, 200000]:
        save_path = os.path.join(get_root_path(), f'checkpoints/xiaorong/{jf_game_id}', f'{jf_game_id}_trgt_cls_pretrain_data_active_th_seed_42(20)_with_teacher')
        train(jf_game_id, 805, 825, 826, 829,
            save_path=save_path, features=ACTIVE_FEATURES, minus_label_col=None, w_t=wt, max_row=None)

