from torch_geometric.data import InMemoryDataset
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
import pandas as pd
from typing import List

chiral_list = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

bond_type_list = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.UNSPECIFIED
]
hybridization_list = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]

bond_dird_list = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]
def atom_features(atom):
    atom_features = []
    atom_features.append(atom.GetAtomicNum())
    atom_features.append(atom.GetTotalNumHs())
    atom_features.append(atom.GetFormalCharge())
    atom_features.append(atom.GetTotalValence())
    atom_features.append(1 if atom.GetIsAromatic() else 0)
    atom_features.append(chiral_list.index(atom.GetChiralTag()))
    atom_features.append(hybridization_list.index(atom.GetHybridization()))
    atom_features.append(atom.GetNumRadicalElectrons())

    return atom_features

def bond_features(bond):
    bond_features = []
    bond_features.append(bond_type_list.index(bond.GetBondType()))
    bond_features.append(1 if bond.GetIsConjugated() else 0)
    bond_features.append(1 if bond.IsInRing() else 0)
    bond_features.append(bond_dird_list.index(bond.GetBondDir()))
    return bond_features


def mol_to_graph(mol):
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    x = np.zeros((num_atoms, 8))
    edge_index = np.zeros((2, 2*num_bonds))
    edge_attr = np.zeros((2*num_bonds, 4))
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = atom_features(atom)
    for i, bond in enumerate(mol.GetBonds()):
        edge_index[0][2*i] = bond.GetBeginAtomIdx()
        edge_index[1][2*i] = bond.GetEndAtomIdx()
        edge_index[0][2*i+1] = bond.GetEndAtomIdx()
        edge_index[1][2*i+1] = bond.GetBeginAtomIdx()
        edge_attr[2*i] = bond_features(bond)
        edge_attr[2*i+1] = bond_features(bond)
    return x, edge_index, edge_attr


def mask_mol_to_graph(mol):
    """
    mask the atoms and corresponding bonds one by one to generate the masked graph
    input: mol: rdkit mol object
    output: a list of masked graphs
    """
    mol_len = mol.GetNumAtoms()
    masked_graphs = []
    # map each atom to its index to mask
    for _ in range(mol_len):
        masked_graph = {}
        masked_graph['x'] = []
        masked_graph['edge_index'] = []
        masked_graph['edge_attr'] = []
        masked_graph['smiles'] = []
        masked_graph['mask'] = []
        masked_graph['mask'].append(_)
        for i, atom in enumerate(mol.GetAtoms()):
            if i == _:
                masked_graph['x'].append([0]*8)
            else:
                masked_graph['x'].append(atom_features(atom))
        for i, bond in enumerate(mol.GetBonds()):
            if bond.GetBeginAtomIdx() == _ or bond.GetEndAtomIdx() == _:
                masked_graph['edge_index'].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                masked_graph['edge_attr'].append([0]*4)
            else:
                masked_graph['edge_index'].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                masked_graph['edge_attr'].append(bond_features(bond))
        masked_graphs.append(masked_graph)
    return masked_graphs

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffold(dataset, include_chirality=False):
    scaffolds = {}
    data_len = len(dataset)
    print(f'Processing {data_len} molecules')
    for i, smiles in enumerate(tqdm(dataset.smiles)):
        scaffold = _generate_scaffold(smiles, include_chirality)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [i]
        else:
            scaffolds[scaffold].append(i)
    # sort the scaffold by the number of molecules in the scaffold
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    all_scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    return all_scaffold_sets

def scaffold_split(dataset, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffold(dataset)
 
    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []
 
    print(f'Sorting {len(dataset)} molecules by scaffold size')
    for scaffold_set in tqdm(scaffold_sets):
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


class FreeSolvDataset(InMemoryDataset):
    def __init__(self, root='data/FreeSolv', transform=None, pre_transform=None):   
        super(FreeSolvDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    
    @property
    def raw_file_names(self):
        return ['freesolv.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self, smiles_line = 'smiles', target_line = 'expt'):
        if self.raw_paths[0].endswith('.csv'):
            df = pd.read_csv(self.raw_paths[0])
            self.smiles = df[smiles_line].tolist()
            self.targets = df[target_line].tolist()
        else:
            raise ValueError('The input file should be a csv file')
        data_list = []
        for i,smiles in enumerate(tqdm(self.smiles)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            x, edge_index, edge_attr = mol_to_graph(mol)
            y = self.targets[i]
            data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), 
                        edge_attr=torch.tensor(edge_attr, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), smiles=smiles)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    from torch.utils.data import Subset
    from model import MPNNPredictor
    dataset = FreeSolvDataset()
    # test mask_mol_to_graph
    mol = Chem.MolFromSmiles(dataset.smiles[0])
    masked_graphs = mask_mol_to_graph(mol)
    print(masked_graphs[0])
    model = MPNNPredictor(node_in_feats=8, edge_in_feats=4)
    x = torch.tensor(masked_graphs[0]['x'], dtype=torch.float)
    x.t().contiguous()
    edge_index = torch.tensor(masked_graphs[0]['edge_index'], dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_attr = torch.tensor(masked_graphs[0]['edge_attr'], dtype=torch.float)
    edge_attr.t().contiguous()
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outs = []
    for _ in range(10):
        out = model(data)
        outs.append(out)
    print(outs)