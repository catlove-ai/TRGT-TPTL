import logging
import os.path

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.trtg_model import TRTGModel
from utils.common import get_root_path
from utils.datasets import MultiLabelDataset, multi_label_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def train(jf_game_id, lr, train_start_date, train_end_date, val_start_date, val_end_date, save_path, features, wise, device='cuda:0', max_row=None, minus_label_col=None):
    # args
    epochs = 10
    batch_size = 1024
    lr = lr
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
    model = TRTGModel(features=features, trgt_layers=1, aux_loss_weight=0.5, wise=wise, device=device)

    # optim
    optimizer = optim.Adam(model.parameters(), lr)
    # scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, min_lr=1e-6)
    scheduler = None
    early_stopping = EarlyStopping(patience=early_stop_patience)

    # data
    train_dataset = MultiLabelDataset(jf_game_id, train_start_date, train_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_label_collate)
    val_dataset = MultiLabelDataset(jf_game_id, val_start_date, val_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_label_collate)

    # train
    trainer = Trainer(model, optimizer, scheduler, early_stopping)
    trainer.train(save_path, epochs, train_loader, val_loader)


if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    jf_game_id = 'wm'
    for jf_game_id in ['wm', 'wxsh', 'mdnf', 'dhp']:
        if jf_game_id == 'mdnf':
            dates = (812, 825, 826, 827)
            lr = 0.001
        elif jf_game_id == 'wxsh':
            dates = (805, 825, 826, 829)
            lr = 0.005
        elif jf_game_id == 'wm':
            dates = (805, 825, 826, 829)
            lr = 0.005
        elif jf_game_id == 'dhp':
            dates = (819, 825, 826, 826)
            lr = 0.001

        aux_loss_weight = 0.5
        for wise in ['none', 'channel']:
            save_path = os.path.join(get_root_path(), f'checkpoints/wise/{jf_game_id}', f'{jf_game_id}_trgt_1_layer_auxloss{aux_loss_weight}_wise_{wise}_active_th_seed_{seed}')
            train(jf_game_id, lr=lr, train_start_date=dates[0], train_end_date=dates[1], val_start_date=dates[2], val_end_date=dates[3],
                  save_path=save_path, features=ACTIVE_FEATURES, wise=wise, minus_label_col=None, max_row=None)

