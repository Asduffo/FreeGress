import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils

from src.guidance.node_model import QM9NodeModel


warnings.filterwarnings("ignore", category=PossibleUserWarning)

@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    print("CFG = ", cfg)
    dataset_config = cfg["dataset"]

    print("get_num_threads", torch.get_num_threads())
    torch.set_num_threads(cfg.train.num_interops_workers)
    torch.set_num_interop_threads(cfg.train.num_interops_workers)
    print("get_num_threads", torch.get_num_threads())

    seed_everything(cfg.train.seed)

    if(isinstance(cfg.general.gpus, int)):
        gpus_ok = cfg.general.gpus >= 0
    elif(isinstance(cfg.general.gpus, omegaconf.ListConfig)):
        gpus_ok = True
        for gpu in cfg.general.gpus:
            if(gpu < 0): gpus_ok = False
    else:
        gpus_ok = False
    
    print("cfg.general.gpus", cfg.general.gpus)
    print("torch.cuda.is_available()", torch.cuda.is_available())

    use_gpu = torch.cuda.is_available() and gpus_ok
    print("gpus_ok", gpus_ok)
    print("use_gpu", use_gpu)


    if dataset_config["name"] in ['sbm', 'comm-20', 'planar']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm-20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)


    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses', 'zinc250k']:
        if dataset_config["name"] == 'qm9':
            print("using QM9")
            from src.datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

            input_size = len(cfg.guidance.guidance_target)

            model = QM9NodeModel(cfg=cfg, dataset=datamodule,  dataset_infos = dataset_infos, input_size=input_size)
            
        elif dataset_config['name'] == 'guacamol':
            from src.datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)

        elif dataset_config.name == 'moses':
            from src.datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
        elif dataset_config.name == 'zinc250k':
            from src.datasets import zinc250k_dataset
            datamodule = zinc250k_dataset.ZINC250KDataModule(cfg)
            dataset_infos = zinc250k_dataset.ZINC250Kinfos(datamodule, cfg)
            train_smiles = None

            input_size = len(cfg.guidance.guidance_target)
            
            model = QM9NodeModel(cfg=cfg, dataset_infos=dataset_infos,
                                 input_size=input_size)
        else:
            raise ValueError("Dataset not implemented")
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    utils.create_folders(cfg)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/loss',
                                              save_last=True,
                                              save_top_k=1,    # was 5
                                              mode='min',
                                              every_n_epochs=cfg.train.ckpt_every_n_train_steps,
                                              save_on_train_epoch_end = True)
        callbacks.append(checkpoint_callback)
    
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      
                      max_epochs=cfg.train.n_epochs,
                      
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=True,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
