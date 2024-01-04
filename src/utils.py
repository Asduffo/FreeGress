import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y, guidance = None):
        self.X = X
        self.E = E
        self.y = y
        self.guidance = guidance

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if(self.guidance is not None):
            self.guidance = self.guidance.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

###############################################################################
# FreeGress stuff
def rstrip1(s, c):
    return s[:-1] if s[-1]==c else s

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors, Crippen

from src.datasets.guided_data import GuidedInMemoryDataset
from src.metrics.properties import penalized_logp, qed
from src.metrics.sascorer import calculateScore
from src.analysis.rdkit_functions import build_molecule
from torch_geometric.data import Batch
import torch.nn.functional as F



def clean_mol(mol):
    if(isinstance(mol, str)):
        mol = Chem.MolFromSmiles(mol)

    Chem.RemoveStereochemistry(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    
    return mol

def graph2mol(data, atom_decoder):
    data = Batch.from_data_list([data])

    #print(data)

    #smonta la variabile "data" costruita sopra e ri-ottiene nodi ed archo
    dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    dense_data = dense_data.mask(node_mask, collapse=True)
    X, E = dense_data.X, dense_data.E

    assert X.size(0) == 1
    atom_types = X[0]
    edge_types = E[0]

    #Questi sono anche i metodi utilizzati quando calcolavamo mu/HOMO in qm9, quindi
    #possiamo fidarci del fatto che funzionino (e comunque sono piuttosto utilizzati
    #in quanto provengono da un paper piuttosto citato da cui hanno preso tutti lo
    #spezzone di codice)
    reconstructed_mol = build_molecule(atom_types, edge_types, atom_decoder)
    return reconstructed_mol

def mol2graph(mol, types, bonds, i, original_smiles = None, estimate_guidance = True):
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()] + 1]

    if len(row) == 0:
        print("Number of rows = 0")
        return None

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    #stime di plogp, mw, sas e logp
    guidance = None
    if(estimate_guidance):
        guidance = torch.zeros((1, 5))
        estimated_plogp = penalized_logp(mol)
        estimated_qed   = qed(original_smiles)
        estimated_mw    = rdMolDescriptors.CalcExactMolWt(mol)
        estimated_sas   = calculateScore(mol)
        estimated_logp  = Crippen.MolLogP(mol)
        
        guidance[0, 0] = estimated_plogp
        guidance[0, 1] = estimated_qed
        guidance[0, 2] = estimated_mw
        guidance[0, 3] = estimated_sas
        guidance[0, 4] = estimated_logp
    
    #questo è l'oggetto effettivo che viene poi usato durante il
    #training. Più in basso verrà salvato in un formato gradito da
    #pytorch per poter essere riutilizzato più volte
    return GuidedInMemoryDataset(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                                y=y, idx=i, guidance=guidance, original_smiles = original_smiles)