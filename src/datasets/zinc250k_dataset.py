from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdMolDescriptors, Crippen
from src.metrics.properties import *
from src.metrics.sascorer import calculateScore

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Batch
from torch_geometric.utils import subgraph
import pandas as pd

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, build_molecule, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule

from rdkit.Chem.MolStandardize import rdMolStandardize
from src.datasets.guided_data import GuidedInMemoryDataset

from src.utils import rstrip1, clean_mol, graph2mol, mol2graph

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value    
    else:
        return [value]

class RemoveYTransform:
    def __call__(self, data):
        data.guidance = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectPenalizedLogPTransform:
    def __call__(self, data):
        data.guidance = data.guidance[..., 0:1]
        return data

class SelectQEDTransform:
    def __call__(self, data):
        data.guidance = data.guidance[..., 1:2]
        return data
    
class SelectMWTransform:
    def __call__(self, data):
        data.guidance = data.guidance[..., 2:3] / 100
        return data

class SelectSASTransform:
    def __call__(self, data):
        data.guidance = data.guidance[..., 3:4]
        return data

class SelectLogPTransform:
    def __call__(self, data):
        
        data.guidance = data.guidance[..., 4:5]
        return data

atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'N+1', 'O-1']

class ZINC250KDataset(InMemoryDataset):
    raw_url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    #raw_url = "https://github.com/aspuru-guzik-group/chemical_vae/blob/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

    def __init__(self, stage, atom_decoder, root, filter_dataset: bool, transform=None, pre_transform=None, 
        pre_filter=None, target_prop=None, remove_h = False, cfg = None):
        self.stage          = stage
        self.atom_decoder   = atom_decoder
        self.filter_dataset = filter_dataset
        self.target_prop    = target_prop
        self.remove_h       = remove_h
        self.cfg            = cfg

        self.types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        #return ['train_zinc250k.csv', 'val_zinc250k.csv', 'test_zinc250k.csv']
        return ['250k_rndm_zinc_drugs_clean_3.csv']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']
        
    def download(self):
        """
        Download raw zinc files
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            #extract_zip(file_path, self.raw_dir)
            #os.unlink(file_path)
        except ImportError:
            print("Failed to download the dataset...")

        if files_exist(self.split_paths):
            return

        #dataset = pd.read_csv(self.raw_paths[1])
        dataset = pd.read_csv(file_path)

        #dataset = dataset.head(1000)

        n_samples = len(dataset)
        n_train = int(0.8 * n_samples)
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        #n_train = 1000
        #n_test = 100
        #n_val = 100

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))
    
    def process(self):
        RDLogger.DisableLog('rdApp.*')
        
        if(self.file_idx != 0):
            root_dir                 = pathlib.Path(os.path.realpath(__file__)).parents[2]
            train_path               = "/data/zinc250k/zinc250k_pyg/raw/new_train.smiles"
            smiles_path              = str(root_dir) + train_path
            my_file                  = open(smiles_path, "r")
            data                     = my_file.read()
            train_smiles             = data.split("\n")

        path = self.split_paths[self.file_idx]
        target_df = pd.read_csv(path)
        
        smiles_list = target_df['smiles'].values
        
        data_list       = []
        smiles_kept     = []

        original_smiles = []

        unmatches = 0
        exceptions = 0
        string_unmatches = 0
        n_already_present = 0

        print("Initial size: ", len(smiles_list))

        build_with_charges = self.cfg.guidance.build_with_partial_charges != "no"
        #Qui è dove il VERO preprocessing avviene. Essenzialmente prende gli smiles
        #del training/val/test set (dipende da chi chiama il metodo) e processa
        #lo smiles per ottenere grafo + guida
        for i, original_smile in enumerate(tqdm(smiles_list)):
            #rimuove \n
            smiles_2D = rstrip1(original_smile, "\n")

            #otteniamo la molecola dallo smiles_2D ed eseguiamo il preprocessing
            mol = clean_mol(smiles_2D, (build_with_charges == False))
            smiles_2D = Chem.MolToSmiles(mol)

            data = mol2graph(mol, self.types, self.bonds, i, smiles_2D, True, build_with_charges)

            #questo succede se abbiamo un grafo vuoto (può succedere)
            #o se un atomo conteneva cariche non neutrali quando non
            #volevamo tenerne
            if(data == None):
                #print("Data is None")
                continue

            #controllo sanità del grafo generato. Qui viene effettivamente 
            #ricostruito il Chem.RWMol della molecola usando la variabile "data" 
            #sopra contenente il grafo della molecola e ne controlla la qualità 
            #(es: se si è spezzata o meno.). Se passa i test, finisce nel training/val/test set.
            #=> PERFETTO PER VEDERE SE LOGP ecc CAMBIANO UNA VOLTA RIASSEMBLATA LA MOLECOLA
            #   PARTENDO DAL GRAFO.
            if self.filter_dataset:
                reconstructed_mol        = graph2mol(data, self.atom_decoder)

                try:
                    reconstructed_mol    = clean_mol(reconstructed_mol, (build_with_charges == False))
                    reconstructed_smiles = Chem.MolToSmiles(reconstructed_mol)
                except:
                    exceptions = exceptions + 1
                    continue

                #Skips if an element of the test set was also in the training set
                if(self.file_idx != 0):
                    if(reconstructed_smiles in train_smiles):
                        n_already_present += 1
                        continue

                ###############################################################
                #se gli smiles della molecola ricostruita e quello originale
                #sono identici, non è neanche necessario controllarne le proprietà.
                #Se differiscono, potrebbero differirne anche le proprietà. Vediamo un po...
                if(smiles_2D == reconstructed_smiles):
                    tmp_mol_original_smile       = mol
                    tmp_mol_smiles_2D            = reconstructed_mol
                    print_smiles                 = False

                    tmp_mol_original_smile_qed   = qed(tmp_mol_original_smile)
                    tmp_mol_original_smile_logp  = Crippen.MolLogP(tmp_mol_original_smile)
                    tmp_mol_original_smile_mw    = rdMolDescriptors.CalcExactMolWt(tmp_mol_original_smile)

                    tmp_mol_smiles_2D_qed        = qed(tmp_mol_smiles_2D)
                    tmp_mol_smiles_2D_logp       = Crippen.MolLogP(tmp_mol_smiles_2D)
                    tmp_mol_smiles_2D_mw         = rdMolDescriptors.CalcExactMolWt(tmp_mol_smiles_2D)

                    if(abs(tmp_mol_original_smile_qed - tmp_mol_smiles_2D_qed) > 1e-5):
                        print_smiles=True
                    if(abs(tmp_mol_original_smile_logp - tmp_mol_smiles_2D_logp) > 1e-5):
                        print_smiles=True
                    if(abs(tmp_mol_original_smile_mw - tmp_mol_smiles_2D_mw) > 4):
                        print_smiles=True
                    
                    if(print_smiles):
                        reconstructed_smiles = None
                        unmatches            = unmatches + 1
                else:
                    string_unmatches = string_unmatches + 1

                ###############################################################

                if reconstructed_smiles is not None and smiles_2D == reconstructed_smiles: 
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(reconstructed_mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(reconstructed_smiles)

                            #DEBUG
                            original_smiles.append(reconstructed_smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        print("Total unmatches: ", unmatches)
        print("exceptions =", exceptions)
        print("string_unmatches: ", string_unmatches)
        print("removed because already present in the training set: ", n_already_present)

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.stage}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")


class ZINC250KDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h       = cfg.dataset.remove_h
        self.datadir        = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.atom_decoder   = atom_decoder
        self.train_smiles   = []

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        
        target = getattr(cfg.guidance, 'guidance_target', None)
        if target == 'penalizedlogp':
            transform = SelectPenalizedLogPTransform()
        elif target == 'qed':
            transform = SelectQEDTransform()
        elif target == 'mw':
            transform = SelectMWTransform()
        elif target == 'sas':
            transform = SelectSASTransform()
        elif target == 'logp':
            transform = SelectLogPTransform()
        elif target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()

        datasets = {'train': ZINC250KDataset(stage='train', atom_decoder=self.atom_decoder, root=root_path, remove_h=cfg.dataset.remove_h,
                                        target_prop=target, transform=transform, filter_dataset = self.filter_dataset, cfg=cfg),
                    'val': ZINC250KDataset(stage='val', atom_decoder=self.atom_decoder, root=root_path, remove_h=cfg.dataset.remove_h,
                                      target_prop=target, transform=transform, filter_dataset = self.filter_dataset, cfg=cfg),
                    'test': ZINC250KDataset(stage='test', atom_decoder=self.atom_decoder, root=root_path, remove_h=cfg.dataset.remove_h,
                                       target_prop=target, transform=transform, filter_dataset = self.filter_dataset, cfg=cfg)}
        super().__init__(cfg, datasets)

class ZINC250Kinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.cfg = cfg

        self.name = 'ZINC250K'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.num_atom_types = len(self.atom_decoder)
        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')

        self.types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        #######################################################################
        #ZINC250k WITH filters (as in the other papers)
        """
        self.num_atom_types = 10
        self.max_weight = 3000

        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2]

        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07}

        self.node_types = torch.tensor([7.3872e-01, 1.1761e-01, 9.9755e-02, 1.4632e-02, 0.0000e+00, 2.3530e-03,
                                        7.8376e-03, 1.7065e-04, 2.2523e-05, 1.8896e-02])

        self.edge_types = torch.tensor([9.0655e-01, 4.8466e-02, 6.0099e-03, 2.4747e-04, 3.8723e-02])

        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                     1.6325e-05, 2.7208e-05, 8.1623e-05, 3.4282e-04, 8.4344e-04, 3.1126e-03,
                                     5.1912e-03, 7.5964e-03, 1.1895e-02, 1.8044e-02, 2.6114e-02, 3.5528e-02,
                                     4.7516e-02, 5.8600e-02, 7.1056e-02, 8.1672e-02, 7.4647e-02, 8.3550e-02,
                                     9.1527e-02, 9.0237e-02, 7.6486e-02, 6.2186e-02, 3.9451e-02, 3.1545e-02,
                                     2.4563e-02, 1.9410e-02, 1.4959e-02, 1.0491e-02, 7.1937e-03, 4.0540e-03,
                                     1.5291e-03, 5.2783e-04, 5.4416e-06])

        self.valency_distribution = torch.tensor([0.0000, 0.1145, 0.2900, 0.3536, 0.2333, 0.0033, 0.0053, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000])
        """
        #######################################################################
        
        self.max_weight = 3000

        #TODO: maybe the last 2 are wrong (since they are 'N+' and 'O-' )
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 4, 1]

        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07, 11: 14, 12: 16}
        
        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                     1.4545e-05, 2.4241e-05, 7.2723e-05, 3.0059e-04, 7.5632e-04, 2.9816e-03,
                                     4.8821e-03, 7.2965e-03, 1.1524e-02, 1.7788e-02, 2.5860e-02, 3.5537e-02,
                                     4.7876e-02, 5.9385e-02, 7.2107e-02, 8.2778e-02, 7.5113e-02, 8.4087e-02,
                                     9.1970e-02, 9.0656e-02, 7.6320e-02, 6.1994e-02, 3.9241e-02, 3.0810e-02,
                                     2.3979e-02, 1.8874e-02, 1.4637e-02, 1.0162e-02, 6.9377e-03, 3.9707e-03,
                                     1.5466e-03, 5.1391e-04, 4.8482e-06])

        self.node_types = torch.tensor([7.3843e-01, 1.0531e-01, 9.6436e-02, 1.4098e-02, 0.0000e+00, 2.2611e-03,
                         7.5426e-03, 1.6223e-04, 2.2906e-05, 1.7853e-02, 1.3654e-02, 4.2335e-03])

        self.edge_types = torch.tensor([9.0644e-01, 4.9232e-02, 5.9592e-03, 2.4122e-04, 3.8128e-02])

        self.valency_distribution = torch.tensor([0.0000, 0.1156, 0.2933, 0.3520, 0.2311, 0.0032, 0.0048, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                  0.0000, 0.0000, 0.0000, 0.0000])
        #######################################################################

        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        
        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
            print("self.max_n_nodes", self.max_n_nodes)
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # after we can be sure we have the data, complete infos
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_zinc_smiles(atom_decoder, train_dataloader, remove_h, 
                                           cfg.guidance.build_with_partial_charges)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_zinc_smiles(atom_decoder, train_dataloader, remove_h, use_partial_charges):
    '''

    :param dataset_name: ZINC-250k
    :return:
    '''
    print(f"\tConverting ZINC-250k dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            if(use_partial_charges == "old_method"):
                mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            else:
                mol = build_molecule(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting ZINC-250k dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

"""
import hydra
import omegaconf
from omegaconf import DictConfig
@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    #ds = [ZINC250KDataset(s, os.path.join(os.path.abspath(__file__), "../../data/zinc250k")) for s in ["train", "val", "test"]]
    print(cfg)

    cfg.dataset.name                        = 'zinc250k' 
    cfg.dataset.datadir                     = 'data/zinc250k/zinc250k_pyg/'
    cfg.dataset.remove_h                    =  True
    cfg.dataset.random_subset               = None
    cfg.dataset.pin_memory                  = False
    cfg.dataset.filter                      = True
    cfg.guidance.build_with_partial_charges = "new_method"

    datamodule = ZINC250KDataModule(cfg)
    dataset_infos = ZINC250Kinfos(datamodule, cfg, recompute_statistics = True)

if __name__ == '__main__':
    main()
"""