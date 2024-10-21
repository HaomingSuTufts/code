#!/cluster/home/hsu02/.conda/envs/hfe/bin/python
#SBATCH --job-name=MPNN
#SBATCH --output=MPNN.out
#SBATCH --error=MPNN.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=dinglab



import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
'''
sys.path.append('/cluster/tufts/dinglab/hsu02/code/DMPNN')
'''

from model import MPNNPredictor
from dataset import FreeSolvDataset
from dataset import scaffold_split, mask_mol_to_graph
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler
import lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from rdkit import Chem

import tensorboard as tb
from torchmetrics import MeanSquaredError


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility
    Args:
    seed: int
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def mc_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''
    A new loss function for capturing the uncertainty of the model
    Args:
    y_pred: (batch_size, mc_iteration)
    y_true: (batch_size, 1)
    
    '''
    mean_pred = torch.mean(y_pred, dim=1)
    var_pred = torch.var(y_pred, dim=1)
    loss_fn  = torch.nn.GaussianNLLLoss()
    loss = loss_fn(mean_pred, y_true.view(-1), var_pred)
    return loss



class predict_model(pl.LightningModule):
    def __init__(self, model, mc_iteration: int = 10, test_task='explain', *args, **kwargs):
        super(predict_model, self).__init__()
        self.model = model
        self.mc_iteration = mc_iteration
        self.test_task = test_task
        self.train_losses = []
        self.val_losses = []
        self.val_rmses = []
        self.val_vars = []

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        pred = self.model(data)
        loss = F.mse_loss(pred, data.y.view(-1, 1))
        self.train_losses.append(loss)
    
        return loss
    
    '''
    def training_step(self, batch, batch_idx):
        outputs = []
        for _ in range(self.mc_iteration):
            data = batch
            pred = self.model(data)
            outputs.append(pred)
        outputs = torch.stack(outputs, dim=1)
        loss = mc_loss(data.y.view(-1, 1), outputs)
        self.log("train_loss", loss, batch_size=data.y.size(0))
        return loss
    '''    

    def on_train_epoch_end(self) -> None:
        train_loss = torch.mean(torch.stack(self.train_losses))
        self.log("train_loss", train_loss)
        self.train_losses.clear()
        torch.save(self.model.state_dict(), 'model.pth')
        return super().on_train_epoch_end()
   
   
    
    def validation_step(self, batch, batch_idx):
        self.model.dropout.train()
        outputs = []
        for _ in range(self.mc_iteration):
            data = batch
            pred = self.model(data)
            outputs.append(pred)
        outputs = torch.stack(outputs, dim=1)
        mean_pred = torch.mean(outputs, dim=1)
        var_pred = torch.var(outputs, dim=1)
        loss = F.mse_loss(mean_pred, data.y.view(-1, 1))
        self.val_losses.append(loss)
        mse = MeanSquaredError().to(self.device)
        rmse = torch.sqrt(mse(mean_pred, data.y.view(-1, 1)))
        self.val_rmses.append(rmse)
        self.val_vars.append(torch.mean(var_pred))
        return loss
    
    def on_validation_epoch_end(self) -> None:
        val_loss = torch.mean(torch.stack(self.val_losses))
        val_rmse = torch.mean(torch.stack(self.val_rmses))
        val_var = torch.mean(torch.stack(self.val_vars))

        metric = val_rmse/val_var
        self.log("val_metric", metric)
        self.log("val_loss", val_loss)
        self.log("val_rmse", val_rmse)
        self.log("val_var", val_var)
     

        self.val_losses.clear()
        self.val_rmses.clear()
        self.val_vars.clear()
        return super().on_validation_epoch_end()
    
    """    
    def validation_step(self, batch, batch_idx):
        self.model.dropout.train()
        outputs = []
        for _ in range(self.mc_iteration):
            data = batch
            pred = self.model(data)
            outputs.append(pred)
        outputs = torch.stack(outputs, dim=1)
        loss = mc_loss(data.y.view(-1, 1), outputs)
        self.log("val_loss", loss, batch_size=data.y.size(0))
        rmse = torch.sqrt(F.mse_loss(data.y.view(-1, 1), torch.mean(outputs, dim=1)))
        self.log("val_rmse", rmse, batch_size=data.y.size(0))
        var = torch.var(outputs, dim=1)
        var = torch.mean(var)
        self.log("val_var", var, batch_size=data.y.size(0))
        return loss
    """

    def test_step(self, batch, batch_idx):
        if self.test_task == 'explain':
            #load the model
            self.model.load_state_dict(torch.load('model.pth'))
            # set the model to eval mode
            self.model.eval()
            # get smiles in each data in the batch
            smiles_list = [data[-1] for data in batch]
            # generate mask for each smiles
            mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            masked_graphs = [mask_mol_to_graph(mol) for mol in mols]
            # get the prediction for each masked graph
            outputs_means = []
            outputs_vars = []
            for masked_graph in masked_graphs:
                
                data = Data(x=torch.tensor(masked_graph['x'], dtype=torch.float).t().contiguous(),
                            edge_index=torch.tensor(masked_graph['edge_index'], dtype=torch.long).t().contiguous(),
                            edge_attr=torch.tensor(masked_graph['edge_attr'], dtype=torch.float).t().contiguous())

                out_puts = []    
                for _ in range(self.mc_iteration):
                    pred = self.model(data)
                    out_puts.append(pred)
                out_puts = torch.stack(out_puts, dim=1)
                outputs_means.append(torch.mean(out_puts, dim=1))
                outputs_vars.append(torch.var(out_puts, dim=1))
            return outputs_means, outputs_vars, smiles_list
        
        elif self.test_task == 'predict':
            # load the model
            self.model.load_state_dict(torch.load('model.pth'))
            # set the model to eval mode
            self.model.eval()
            data = batch
            outputs = []
            for _ in range(self.mc_iteration):
                pred = self.model(data)
                outputs.append(pred)
            outputs = torch.stack(outputs, dim=1)
            return outputs, data.y.view(-1, 1)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer
    
    
  

def split_dataset(dataset: torch_geometric.data.Dataset, split: float = 0.8,split_type: str = 'random'):
    if split_type == 'random':
        val_size = test_size = (1 - split) / 2
        train_size = split
        num_data = len(dataset)
        indices = list(range(num_data))
        np.random.shuffle(indices)
        split1 = int(np.floor(train_size * num_data))
        split2 = int(np.floor((train_size + val_size) * num_data))
        train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
        return train_indices, val_indices, test_indices
    
    elif split_type == 'scaffold':
        val_size = test_size = (1 - split) / 2
        train_indices, val_indices, test_indices = scaffold_split(dataset, val_size, test_size)
        return train_indices, val_indices, test_indices
    
def main(args):
    torch.set_float32_matmul_precision('high')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FreeSolvDataset()
    train_indices, val_indices, test_indices = split_dataset(dataset, split=0.8, split_type=args.split_type)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=args.bath_size, sampler=train_sampler,
                               num_workers=10, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(dataset, batch_size=args.bath_size, sampler=val_sampler, 
                            num_workers=10, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset, batch_size=args.bath_size, sampler=test_sampler, 
                             num_workers=10, pin_memory=True, persistent_workers=True)

    gnn = MPNNPredictor(node_in_feats=8, edge_in_feats=4, node_out_feats=300, edge_hidden_feats=256, num_step_message_passing=6)
    gnn = gnn.to(device)

    model = predict_model(gnn, mc_iteration=args.mc_iteration)
    trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split_type", type=str, default='random')
    argparser.add_argument("--seed", type=int, default=114)
    argparser.add_argument("--epochs", type=int, default=100)    
    argparser.add_argument("--mc_iteration", type=int, default=1)
    argparser.add_argument("--bath_size", type=int, default=32)
    args = argparser.parse_args()
    main(args)

