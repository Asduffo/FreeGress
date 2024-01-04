import torch
import time

import pytorch_lightning as pl

from src import utils

from torch_geometric.data.lightning import LightningDataset

from src.datasets.abstract_dataset import AbstractDatasetInfos

class NodeModel(pl.LightningModule):
    def __init__(self, 
                 cfg,
                 dataset_infos : AbstractDatasetInfos,
                 input_size : int):
        super().__init__()

        self.cfg = cfg
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.vl_loss    = torch.nn.CrossEntropyLoss()
        
        self.dataset_infos = dataset_infos

        self.output_dim = dataset_infos.max_n_nodes + 1 #+1 to account for zero
        self.input_dim  = input_size

        self.initialize_model()

    def initialize_model(self):
        pass

    def training_step(self, X):
        pass

    def calculate_target(self, data):
        #print("data.batch", data.batch[:200])

        to_return = torch.bincount(data.batch).long().to(self.device)
        #print("to_return", to_return)

        to_return = torch.nn.functional.one_hot(to_return, num_classes = self.output_dim).float()

        #torch.set_printoptions(profile="full")
        #print("to_return", to_return[:3])
        #print(torch.argmax(to_return, dim=1)[:3])

        to_return = torch.argmax(to_return, dim=1)

        return to_return
    
    def forward_data(self, X):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

class QM9NodeModel(NodeModel):
    def __init__(self, 
                 cfg, 
                 dataset_infos : AbstractDatasetInfos,
                 input_size : int):
        super().__init__(cfg=cfg, dataset_infos=dataset_infos, input_size=input_size)

    def initialize_model(self):
        self.layer_in  = torch.nn.Linear(self.input_dim, 512).to(self.device)
        self.layer_01  = torch.nn.Linear(512, 512).to(self.device)
        self.layer_out = torch.nn.Linear(512, self.output_dim).to(self.device)

    def forward_data(self, X):
        X = self.layer_in(X)
        X = torch.nn.functional.relu(X)

        X = self.layer_01(X)
        X = torch.nn.functional.relu(X)

        X = self.layer_out(X)
        #X = torch.nn.functional.softmax(X)
        
        return X

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        #self.train_loss.reset()

    def training_step(self, data, i):
        #saves into a list the number of nodes in each element
        target = self.calculate_target(data)

        out = self.forward_data(data.guidance)
        loss = self.train_loss(out, target)

        self.log("loss", value = loss.item(), prog_bar = True)
        return {'loss': loss}
    
    ###########################################################################

    def on_validation_epoch_start(self) -> None:
        self.vl_step_count = 0
        self.vl_loss_sum = 0

    def validation_step(self, data, i):
        #saves into a list the number of nodes in each element
        target = self.calculate_target(data)

        out = self.forward_data(data.guidance)
        vl_loss = self.vl_loss(out, target)

        #print("out", out.argmax(dim=1)[:15])
        #print("target", target[:15])

        self.log("val/loss", value = vl_loss.item(), prog_bar = True, batch_size=self.cfg.train.batch_size)
        self.vl_step_count = self.vl_step_count + 1
        self.vl_loss_sum = self.vl_loss_sum + vl_loss.item()
        
        return {'val/loss': vl_loss}
    
    def on_validation_epoch_end(self):
        average_vl_loss = self.vl_loss_sum / self.vl_step_count
        print("Epoch mean average validation loss: ", average_vl_loss)
        self.log("val/loss", average_vl_loss, sync_dist=True)

    ###########################################################################

    def on_test_epoch_start(self):
        self.ts_step_count = 0
        self.ts_loss_sum = 0

    def test_step(self, data, i):
        #saves into a list the number of nodes in each element
        target = self.calculate_target(data)


        out = self.forward_data(data.guidance)
        ts_loss = self.vl_loss(out, target)

        self.log("test/loss", value = ts_loss.item(), prog_bar = True, batch_size=self.cfg.train.batch_size)
        self.ts_step_count = self.ts_step_count + 1
        self.ts_loss_sum   = self.ts_loss_sum   + ts_loss.item()
        return {'test/loss': ts_loss}
    
    def on_test_epoch_end(self):
        average_ts_loss = self.ts_loss_sum / self.ts_step_count
        print("Epoch mean test loss: ", average_ts_loss)
        self.log("test/loss", average_ts_loss, sync_dist=True)