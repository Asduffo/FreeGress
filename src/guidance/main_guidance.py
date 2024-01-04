import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import psi4
from rdkit import Chem
import torch
import wandb
import hydra
import os
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings


import src.utils as utils
from src.guidance.guidance_diffusion_model_discrete import DiscreteDenoisingDiffusionUnconditional
from src.datasets import qm9_dataset, zinc250k_dataset
from src.metrics.molecular_metrics import SamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import  TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.utils import update_config_with_new_keys
from src.guidance.qm9_regressor_discrete import Qm9RegressorDiscrete

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):

    saved_cfg                   = cfg.copy()
    name                        = cfg.general.name + '_resume'
    resume                      = cfg.general.test_only
    final_samples_to_generate   = cfg.general.final_model_samples_to_generate
    final_chains_to_save        = cfg.general.final_model_chains_to_save

    batch_size                  = cfg.train.batch_size
    n_epochs                    = cfg.train.n_epochs
    n_test_molecules_to_sample  = cfg.guidance.n_test_molecules_to_sample
    n_samples_per_test_molecule = cfg.guidance.n_samples_per_test_molecule
    node_model_path             = cfg.guidance.node_model_path
    build_with_partial_charges  = cfg.guidance.build_with_partial_charges
    experiment_type             = cfg.guidance.experiment_type
    guidance_properties_list    = cfg.guidance.guidance_properties_list
    test_thresholds             = cfg.guidance.test_thresholds
    wandb                       = cfg.general.wandb
    gpus                        = cfg.general.gpus
    lambda_guidance             = cfg.guidance.lambda_guidance 
    include_split               = cfg.guidance.include_split
    
    model = DiscreteDenoisingDiffusionUnconditional.load_from_checkpoint(resume, **model_kwargs)

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.general.final_model_samples_to_generate = final_samples_to_generate
        cfg.general.final_model_chains_to_save   = final_chains_to_save
        cfg.train.batch_size                     = batch_size
        cfg.train.n_epochs                       = n_epochs
        cfg.guidance.n_test_molecules_to_sample  = n_test_molecules_to_sample
        cfg.guidance.n_samples_per_test_molecule = n_samples_per_test_molecule 
        cfg.guidance.node_model_path             = node_model_path
        cfg.guidance.build_with_partial_charges  = build_with_partial_charges
        cfg.guidance.experiment_type             = experiment_type
        cfg.guidance.guidance_properties_list    = guidance_properties_list
        cfg.guidance.test_thresholds             = test_thresholds
        cfg.general.wandb                        = wandb

        cfg.general.gpus                         = gpus
        cfg.guidance.guidance_medium             = "NONE"
        cfg.guidance.lambda_guidance             = lambda_guidance
        cfg.guidance.include_split               = include_split

    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'guidance', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print(cfg)
    
    print("get_num_threads", torch.get_num_threads())
    torch.set_num_threads(cfg.train.num_interops_workers)
    torch.set_num_interop_threads(cfg.train.num_interops_workers)
    print("get_num_threads", torch.get_num_threads())

    seed_everything(cfg.train.seed)

    assert dataset_config.name in ["qm9", "zinc250k"], "Only QM9/ZINC dataset is supported for now"

    if dataset_config["name"] == 'qm9':
        datamodule = qm9_dataset.QM9DataModule(cfg, regressor=True)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        datamodule.prepare_data()
        train_smiles = qm9_dataset.get_train_smiles(cfg, datamodule.train_dataloader(), dataset_infos)
    elif dataset_config["name"] == 'zinc250k':
        datamodule = zinc250k_dataset.ZINC250KDataModule(cfg)
        dataset_infos = zinc250k_dataset.ZINC250Kinfos(datamodule=datamodule, cfg=cfg)
        datamodule.prepare_data()
        train_smiles = None
    else:
        raise ValueError("Dataset not implemented")
    

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features, 'load_model': True}

    # When testing, previous configuration is fully loaded
    cfg_pretrained, guidance_sampling_model = get_resume(cfg, model_kwargs)

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model = cfg_pretrained.model
    model_kwargs['load_model'] = False

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    # load pretrained regressor
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    guidance_model = Qm9RegressorDiscrete.load_from_checkpoint(os.path.join(cfg.general.trained_regressor_path))

    model_kwargs['guidance_model'] = guidance_model

    if(isinstance(cfg.general.gpus, int)):
        gpus_ok = cfg.general.gpus >= 0
        n_gpus = cfg.general.gpus
    elif(isinstance(cfg.general.gpus, omegaconf.ListConfig)):
        gpus_ok = True
        n_gpus = 0
        for gpu in cfg.general.gpus:
            if(gpu < 0): gpus_ok = False
            n_gpus = n_gpus + 1
    else:
        gpus_ok = False
        n_gpus = 0
    
    print("cfg.general.gpus", cfg.general.gpus)
    print("torch.cuda.is_available()", torch.cuda.is_available())

    use_gpu = torch.cuda.is_available() and gpus_ok
    print("gpus_ok", gpus_ok)
    print("use_gpu", use_gpu)
    

    if cfg.general.name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      devices=cfg.general.gpus if use_gpu else None,
                      limit_test_batches=cfg.general.test_samples_to_generate,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      enable_progress_bar=False,
                      logger=[],
                      )

    # add for conditional sampling
    model = guidance_sampling_model
    model.args = cfg
    model.guidance_model = guidance_model
    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == '__main__':
    main()
