from typing import Callable, Optional
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

class GuidedInMemoryDataset(Data):
    def __init__(self, x: OptTensor = None, 
                 edge_index: OptTensor = None, 
                 edge_attr: OptTensor = None, 
                 y: OptTensor = None, 
                 pos: OptTensor = None,
                 guidance = None,
                 original_smiles = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.guidance = guidance
        self.original_smiles = original_smiles