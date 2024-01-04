import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils

####################################################à
#extra imports

from torch.distributions import Categorical
from torchmetrics import MeanAbsoluteError
from src.guidance.node_model import QM9NodeModel

from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit import Chem
import math
try:
    import psi4
except ModuleNotFoundError:
    print("PSI4 not found")
from src.analysis.rdkit_functions import build_molecule, mol2smiles, build_molecule_with_partial_charges
import pickle
import pandas as pd

from rdkit.Chem import Crippen
from src.metrics.sascorer import calculateScore
from src.utils import graph2mol, clean_mol
from src.metrics.properties import mw, penalized_logp, qed

from datasets import qm9_dataset

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        ##################################################################################

        #this will be used as the guidance when we need to calculate p(G_{t-1}|G_t) without the guidance
        if(self.cfg.guidance.trainable_cf == True):
            self.cf_null_token = torch.nn.parameter.Parameter(torch.randn(size = (1, self.gdim)))
        else:
            self.cf_null_token = torch.zeros(size = (1, self.gdim))

        # specific properties to generate molecules
        self.cond_val = MeanAbsoluteError()
        self.num_valid_molecules = 0
        self.num_total = 0

        #wish I had a more elegant way to do this
        self.node_model = None

        #stores the generated smiles on the test step
        self.generated_smiles = []

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, data.guidance, train_step = True)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, data.guidance)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False,
                                    guidance = data.guidance)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        if(self.cfg.train.batch_size > 1):
            print("WARNING: batch size > 1. You may not have enough batches to run this test.",
                  "Try relaunching this experiment with train.batch_size=1")
            print("batch size is:", self.cfg.train.batch_size)
            data.guidance = data.guidance[0,:]

        #checks if the current guidance element satisfies the required ranges
        if(self.cfg.guidance.experiment_type == 'accuracy'):
            valid_properties = self.cfg.guidance.guidance_properties_list
            test_thresholds  = self.cfg.guidance.test_thresholds

            k = 0
            for valid_property in valid_properties:
                curr_property = data.guidance[0, k].item()
                valid_threshold = test_thresholds[valid_property]

                if(valid_threshold[0] > curr_property or curr_property > valid_threshold[1]):
                    print(f"passing {i}-th element (value = {curr_property}).")
                    return None

                k = k + 1
        
        print(f'Select No.{i+1} test molecule')
        # Extract properties
        target_properties = data.guidance.clone()
        
        #loads self.node_model (only the first time because of the check self.node_model == None)
        if(self.node_model == None and self.cfg.guidance.node_model_path != None):
            #actually unused (zinc ended up not using guidance_target='both')
            if(self.cfg.guidance.guidance_target == 'both'):
                if(self.cfg.dataset.name == 'zinc250k'):
                    input_size = 2
                else:
                    input_size = 2
            else:
                input_size = 1

            node_model_kwargs  = {'cfg': self.cfg, 'dataset_infos': self.dataset_info, 'input_size': input_size}
            self.node_model = QM9NodeModel.load_from_checkpoint(self.cfg.guidance.node_model_path, **node_model_kwargs)
            self.node_model.cfg.general.gpus = self.cfg.general.gpus

        num_nodes = None
        if(self.node_model != None):
            input_guidance = data.guidance.repeat(self.cfg.guidance.n_samples_per_test_molecule, 1).to(data.guidance.device)
            num_nodes = torch.nn.functional.softmax(self.node_model.forward_data(input_guidance), dim=-1)
            #print("raw num_nodes", num_nodes)
            print("num nodes shape:", num_nodes.size())
            
            if(self.cfg.guidance.node_inference_method == "sample"):
                #this takes num_nodes, and FOR EACH ROW INDEPENDENTLY samples one integer
                #from the distribution represented by that row
                num_nodes = torch.tensor([Categorical(num_nodes[i, ...]).sample().item() 
                                          for i in range(num_nodes.shape[0])]).to(data.guidance.device)
            else:
                num_nodes = torch.argmax(num_nodes, dim = -1).to(data.guidance.device)
            print("num_nodes", num_nodes)

        data.guidance = torch.zeros(data.guidance.shape[0], 0).type_as(data.guidance)
        print("TARGET PROPERTIES", target_properties)

        start = time.time()
        
        ident = 0
        samples = self.sample_batch(batch_id=ident, 
                                    batch_size=self.cfg.guidance.n_samples_per_test_molecule, 
                                    num_nodes=num_nodes,
                                    save_final=10,
                                    keep_chain=1,
                                    number_chain_steps=self.number_chain_steps,
                                    guidance=target_properties)
        print(f'Sampling took {time.time() - start:.2f} seconds\n')
        self.save_cond_samples(samples, target_properties, file_path=os.path.join(os.getcwd(), f'cond_smiles{i}.pkl'))


        #######################################################################
        """
        mol = graph2mol(data, self.dataset_info.atom_decoder)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E
        atom_types = X[0]
        edge_types = E[0]
        samples = [[atom_types, edge_types] for i in range(1)]

        smiles = mol2smiles(mol)

        
        if(data.original_smiles[0] == smiles): print("matching smiles")
        else: print("!!!!!!!!!!!!!!!!!!!!!! unmatching smiles: ", data.original_smiles, " became ", smiles)
        print("original smiles", smiles)
        """
        #######################################################################

        # save conditional generated samples
        mae = self.cond_sample_metric(samples, target_properties)

        print("==============================================================")
        return {'mae': mae}

    def on_test_epoch_end(self) -> None:
        final_mae = self.cond_val.compute()
        final_validity = self.num_valid_molecules / self.num_total
        print("Final MAE", final_mae)
        print("Final validity", final_validity * 100)

        #######################################################################
        unique_generated_smiles = set(self.generated_smiles)
        if(len(self.generated_smiles) != 0):
            final_uniqueness = len(unique_generated_smiles)/len(self.generated_smiles)
            print("final_uniqueness = ", final_uniqueness)
        else:
            final_uniqueness = 0
            print("final_uniqueness = 0 due to no uniques")

        #Old method to get the dataset training smiles
        """
        #opens the training set smiles
        
        #path = os.path.join(os.path.dirname(os.getcwd()), self.cfg.dataset.datadir, "/raw/new_train.smiles")
        path = "/DiGress/" + self.cfg.dataset.datadir + "raw/new_train.smiles"
        print("opening ", path)
        my_file = open(path, "r")
  
        # reading the file
        data                     = my_file.read()
        train_dataset_smiles     = data.split("\n")
        """

        #new method (DELETE ME IF YOU USE THE OLD METHOD)
        if(self.cfg.dataset.name == 'qm9'):
            train_dataset_smiles = qm9_dataset.get_train_smiles(cfg=self.cfg, train_dataloader=None,
                                                        dataset_infos=self.dataset_infos, evaluate_dataset=False)
        else:
            print("TODO: implement get_train_smiles for other datasets")

        train_dataset_smiles_set = set(train_dataset_smiles)
        print("There are ", len(train_dataset_smiles_set), " smiles in the training set")

        """
        unique_generated_smiles = set(["CC(C)C(=O)Nc1ccc(NC(=O)NC(C)C(=O)N2CCCCC2C)cc1",
                                       "KK",
                                       "COc1ccc(-c2nnc(SC3CCOC3=O)n2-c2ccccc2)cc1",
                                       "BB",
                                       "CC(C)n1nc(CC(=O)Nc2cccc3c2ccn3C(C)C)c2ccccc2c1=O"])
        train_dataset_smiles_set = set(["AA", "BB", "CC", "DD", "EE"])
        self.generated_smiles = unique_generated_smiles
        """

        final_novelty_smiles   = unique_generated_smiles.difference(train_dataset_smiles_set)

        if(len(unique_generated_smiles) != 0):
            final_novelty          = len(final_novelty_smiles)/len(unique_generated_smiles)
            print("final_novelty", final_novelty)
        else:
            final_novelty = 0
            print("Final Novelty = 0 due to no smiles generated")
        #######################################################################

        wandb.run.summary['final_MAE'] = final_mae
        wandb.run.summary['final_validity'] = final_validity
        wandb.run.summary['final_uniqueness'] = final_uniqueness
        wandb.log({'final mae': final_mae,
                   'final validity': final_validity,
                   'final uniqueness': final_uniqueness})
        
    def cond_sample_metric(self, samples, input_properties):
        if(self.cfg.guidance.experiment_type in ['new_method', 'accuracy']):
            return self.accuracy_test(samples, input_properties)
        else:
            return self.original_test(samples, input_properties)

    
    def accuracy_test(self, samples, input_properties):
        valid_properties = self.cfg.guidance.guidance_properties_list
        test_thresholds  = self.cfg.guidance.test_thresholds

        if('mu' in valid_properties or 'homo' in valid_properties):
            try:
                import psi4
                # Hardware side settings (CPU thread number and memory settings used for calculation)
                psi4.set_num_threads(nthread=4)
                psi4.set_memory("5GB")
                psi4.core.set_output_file('psi4_output.dat', False)
            except ModuleNotFoundError:
                print("PSI4 not found")

        print("valid_properties", valid_properties)

        numerical_results_dict = {}
        binary_results_dict    = {}
        for tgt in valid_properties:
            numerical_results_dict[tgt] = []
            binary_results_dict[tgt]    = []

        split_molecules = 0
        sample_smiles = []

        for sample in samples:
            if(self.cfg.guidance.build_with_partial_charges):
                raw_mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            else:
                raw_mol = build_molecule(sample[0], sample[1], self.dataset_info.atom_decoder)

            mol = Chem.rdchem.RWMol(raw_mol)

            try:
                Chem.SanitizeMol(mol)
            except:
                print('invalid chemistry')
                continue

            # Coarse 3D structure optimization by generating 3D structure from SMILES
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1
            try:
                EmbedMolecule(mol, params)
            except Chem.rdchem.AtomValenceException:
                print('invalid chemistry')
                continue

            # Structural optimization with MMFF (Merck Molecular Force Field)
            try:
                s = MMFFOptimizeMolecule(mol)
                print(s)
            except:
                print('Bad conformer ID')
                continue

            try:
                conf = mol.GetConformer()
            except:
                print("GetConformer failed")
                continue

            if(self.cfg.guidance.include_split == False):
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if(len(mol_frags) > 1):
                    print("Ignoring a split molecule")
                    split_molecules = split_molecules + 1
                    continue

            ##########################################################
            ##########################################################

            if('mu' in valid_properties or 'homo' in valid_properties):
                # Convert to a format that can be input to Psi4.
                # Set charge and spin multiplicity (below is charge 0, spin multiplicity 1)

                # Get the formal charge
                fc = 'FormalCharge'
                mol_FormalCharge = int(mol.GetProp(fc)) if mol.HasProp(fc) else Chem.GetFormalCharge(mol)

                sm = 'SpinMultiplicity'
                if mol.HasProp(sm):
                    mol_spin_multiplicity = int(mol.GetProp(sm))
                else:
                    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
                    NumRadicalElectrons = 0
                    for Atom in mol.GetAtoms():
                        NumRadicalElectrons += Atom.GetNumRadicalElectrons()
                    TotalElectronicSpin = NumRadicalElectrons / 2
                    SpinMultiplicity = 2 * TotalElectronicSpin + 1
                    mol_spin_multiplicity = int(SpinMultiplicity)

                mol_input = "%s %s" % (mol_FormalCharge, mol_spin_multiplicity)
                print(mol_input)
                #mol_input = "0 1"

                # Describe the coordinates of each atom in XYZ format
                for atom in mol.GetAtoms():
                    mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x) \
                                + " " + str(conf.GetAtomPosition(atom.GetIdx()).y) \
                                + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

                try:
                    molecule = psi4.geometry(mol_input)
                except:
                    print('Can not calculate psi4 geometry')
                    continue

                # Convert to a format that can be input to pyscf
                # Set calculation method (functional) and basis set
                level = "b3lyp/6-31G*"

                # Calculation method (functional), example of basis set
                # theory = ['hf', 'b3lyp']
                # basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

                # Perform structural optimization calculations
                print('Psi4 calculation starts!!!')
                #energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
                try:
                    energy, wave_function = psi4.energy(level, molecule=molecule, return_wfn=True)
                except:
                    print("Psi4 did not converge")
                    continue

                print('Chemistry information check!!!')
            
            ##########################################################
            ##########################################################
            try:
                mol = raw_mol
                mol = clean_mol(mol)

                smile = Chem.MolToSmiles(mol)
                print("Generated SMILES ", smile)
            except:
                print("clean_mol failed")
                continue

            sample_smiles.append(smile)
            self.generated_smiles.append(smile)

            if 'mu' in valid_properties:
                dip_x, dip_y, dip_z = wave_function.variable('SCF DIPOLE')[0],\
                                      wave_function.variable('SCF DIPOLE')[1],\
                                      wave_function.variable('SCF DIPOLE')[2]
                dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                print("Dipole moment", dipole_moment)
                
                thresholds = test_thresholds['mu']
                if(thresholds[0] <= dipole_moment and dipole_moment <= thresholds[1]):
                    binary_results_dict['mu'].append(1)
                else:
                    binary_results_dict['mu'].append(0)
                numerical_results_dict['mu'].append(dipole_moment)

            if 'homo' in valid_properties:
                # Compute HOMO (Unit: au= Hartree）
                LUMO_idx = wave_function.nalpha()
                HOMO_idx = LUMO_idx - 1
                homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]

                # convert unit from a.u. to ev
                homo = homo * 27.211324570273

                thresholds = test_thresholds['homo']
                if(thresholds[0] <= homo and homo <= thresholds[1]):
                    binary_results_dict['homo'].append(1)
                else:
                    binary_results_dict['homo'].append(0)
                numerical_results_dict['homo'].append(homo)

            if 'penalizedlogp' in valid_properties:
                plogp_estimate = penalized_logp(mol)

                thresholds = test_thresholds['penalizedlogp']
                if(thresholds[0] <= plogp_estimate and plogp_estimate <= thresholds[1]):
                    binary_results_dict['penalizedlogp'].append(1)
                else:
                    binary_results_dict['penalizedlogp'].append(0)
                numerical_results_dict['penalizedlogp'].append(plogp_estimate)

            if 'logp' in valid_properties:
                logp_estimate = Crippen.MolLogP(mol)

                thresholds = test_thresholds['logp']
                if(thresholds[0] <= logp_estimate and logp_estimate <= thresholds[1]):
                    binary_results_dict['logp'].append(1)
                else:
                    binary_results_dict['logp'].append(0)
                numerical_results_dict['logp'].append(logp_estimate)

            if 'qed' in valid_properties:
                qed_estimate = qed(mol)

                thresholds = test_thresholds['qed']
                if(thresholds[0] <= qed_estimate and qed_estimate <= thresholds[1]):
                    binary_results_dict['qed'].append(1)
                else:
                    binary_results_dict['qed'].append(0)
                numerical_results_dict['qed'].append(qed_estimate)

            if 'mw' in valid_properties:
                mw_estimate = mw(mol) / 100

                thresholds = test_thresholds['mw']
                if(thresholds[0] <= mw_estimate and mw_estimate <= thresholds[1]):
                    binary_results_dict['mw'].append(1)
                else:
                    binary_results_dict['mw'].append(0)
                numerical_results_dict['mw'].append(mw_estimate)

            if 'sas' in valid_properties:
                sas_estimate = calculateScore(mol)

                thresholds = test_thresholds['sas']
                if(thresholds[0] <= sas_estimate and sas_estimate <= thresholds[1]):
                    binary_results_dict['sas'].append(1)
                else:
                    binary_results_dict['sas'].append(0)
                numerical_results_dict['sas'].append(sas_estimate)

        num_valid_molecules = 0
        outputs = None
        binary_outputs = None

        for tgt in valid_properties:
            num_valid_molecules = max(num_valid_molecules, len(numerical_results_dict[tgt]))

            curr_numerical = torch.FloatTensor(numerical_results_dict[tgt])
            curr_binary    = torch.FloatTensor(binary_results_dict[tgt])

            if(outputs == None):
                outputs        = curr_numerical.unsqueeze(1)
                binary_outputs = curr_binary.unsqueeze(1)
            else:
                print("outputs", outputs)
                print("binary_outputs", binary_outputs)
                outputs        = torch.hstack((outputs, curr_numerical.unsqueeze(1)))
                binary_outputs = torch.hstack((binary_outputs, curr_binary.unsqueeze(1)))
        
        print("Number of valid samples", num_valid_molecules)
        self.num_valid_molecules += num_valid_molecules
        self.num_total += len(samples)

        #we can recycle tgt for outputs[tgt]

        target_tensor = input_properties.repeat(outputs.size(0), 1).cpu()

        print("outputs", outputs)

        mae = self.cond_val(outputs, target_tensor)
        accuracy = torch.mean(torch.count_nonzero(binary_outputs).type(torch.FloatTensor))
        
        unique_smiles              = set(sample_smiles)
        n_unique_smiles            = len(unique_smiles)
        if(len(sample_smiles) != 0):
            n_unique_smiles_percentage = n_unique_smiles/len(sample_smiles)
        else:
            n_unique_smiles_percentage = 0
        print("percentage of unique_samples: ", n_unique_smiles_percentage)
        
        print("binary_outputs =", binary_outputs)
        print("target_tensor  =", target_tensor)

        print('Conditional generation metric:')
        print(f'Epoch {self.current_epoch}: MAE: {mae}')
        print(f'Epoch {self.current_epoch}: success rate: {accuracy}')

        wandb.log({"val_epoch/conditional generation mae": mae,
                   'Valid molecules'                     : num_valid_molecules,
                   'Valid molecules splitted'            : split_molecules,
                   "val_epoch/n_unique_smiles"           : n_unique_smiles,
                   "val_epoch/n_unique_smiles_percentage": n_unique_smiles_percentage,
                   })

        if(self.cfg.guidance.experiment_type == 'accuracy'):
            wandb.log({"val_epoch/accuracy": accuracy})

        return mae


    def original_test(self, samples, input_properties):
        try:
            import psi4
        except ModuleNotFoundError:
            print("PSI4 not found")
        mols_dipoles = []
        mols_homo = []

        # Hardware side settings (CPU thread number and memory settings used for calculation)
        psi4.set_num_threads(nthread=4)
        psi4.set_memory("5GB")
        psi4.core.set_output_file('psi4_output.dat', False)

        for sample in samples:
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)

            try:
                Chem.SanitizeMol(mol)
            except:
                print('invalid chemistry')
                continue

            # Coarse 3D structure optimization by generating 3D structure from SMILES
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1
            try:
                EmbedMolecule(mol, params)
            except Chem.rdchem.AtomValenceException:
                print('invalid chemistry')
                continue

            # Structural optimization with MMFF (Merck Molecular Force Field)
            try:
                s = MMFFOptimizeMolecule(mol)
                print(s)
            except:
                print('Bad conformer ID')
                continue

            try:
                conf = mol.GetConformer()
            except:
                print("GetConformer failed")
                continue

            # Convert to a format that can be input to Psi4.
            # Set charge and spin multiplicity (below is charge 0, spin multiplicity 1)

            # Get the formal charge
            fc = 'FormalCharge'
            mol_FormalCharge = int(mol.GetProp(fc)) if mol.HasProp(fc) else Chem.GetFormalCharge(mol)

            sm = 'SpinMultiplicity'
            if mol.HasProp(sm):
                mol_spin_multiplicity = int(mol.GetProp(sm))
            else:
                # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
                NumRadicalElectrons = 0
                for Atom in mol.GetAtoms():
                    NumRadicalElectrons += Atom.GetNumRadicalElectrons()
                TotalElectronicSpin = NumRadicalElectrons / 2
                SpinMultiplicity = 2 * TotalElectronicSpin + 1
                mol_spin_multiplicity = int(SpinMultiplicity)

            mol_input = "%s %s" % (mol_FormalCharge, mol_spin_multiplicity)
            print(mol_input)
            #mol_input = "0 1"

            # Describe the coordinates of each atom in XYZ format
            for atom in mol.GetAtoms():
                mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x) \
                             + " " + str(conf.GetAtomPosition(atom.GetIdx()).y) \
                             + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

            try:
                molecule = psi4.geometry(mol_input)
            except:
                print('Can not calculate psi4 geometry')
                continue

            # Convert to a format that can be input to pyscf
            # Set calculation method (functional) and basis set
            level = "b3lyp/6-31G*"

            # Calculation method (functional), example of basis set
            # theory = ['hf', 'b3lyp']
            # basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

            # Perform structural optimization calculations
            print('Psi4 calculation starts!!!')
            #energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
            try:
                energy, wave_function = psi4.energy(level, molecule=molecule, return_wfn=True)
            except:
                print("Psi4 did not converge")
                continue

            print('Chemistry information check!!!')

            if self.cfg.guidance.guidance_target in ['mu', 'both']:
                dip_x, dip_y, dip_z = wave_function.variable('SCF DIPOLE')[0],\
                                      wave_function.variable('SCF DIPOLE')[1],\
                                      wave_function.variable('SCF DIPOLE')[2]
                dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                print("Dipole moment", dipole_moment)
                mols_dipoles.append(dipole_moment)

            if self.cfg.guidance.guidance_target in ['homo', 'both']:
                # Compute HOMO (Unit: au= Hartree）
                LUMO_idx = wave_function.nalpha()
                HOMO_idx = LUMO_idx - 1
                homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]

                # convert unit from a.u. to ev
                homo = homo * 27.211324570273
                print("HOMO", homo)
                mols_homo.append(homo)

        num_valid_molecules = max(len(mols_dipoles), len(mols_homo))
        print("Number of valid samples", num_valid_molecules)
        self.num_valid_molecules += num_valid_molecules
        self.num_total += len(samples)

        mols_dipoles = torch.FloatTensor(mols_dipoles)
        mols_homo = torch.FloatTensor(mols_homo)

        if self.cfg.guidance.guidance_target == 'mu':
            mae = self.cond_val(mols_dipoles.unsqueeze(1),
                                input_properties.repeat(len(mols_dipoles), 1).cpu())

        elif self.cfg.guidance.guidance_target == 'homo':
            mae = self.cond_val(mols_homo.unsqueeze(1),
                                input_properties.repeat(len(mols_homo), 1).cpu())

        elif self.cfg.guidance.guidance_target == 'both':
            properties = torch.hstack((mols_dipoles.unsqueeze(1), mols_homo.unsqueeze(1)))
            mae = self.cond_val(properties,
                                input_properties.repeat(len(mols_dipoles), 1).cpu())

        print('Conditional generation metric:')
        print(f'Epoch {self.current_epoch}: MAE: {mae}')
        wandb.log({"val_epoch/conditional generation mae": mae,
                   'Valid molecules': num_valid_molecules})
        return mae


    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        if(self.cfg.guidance.loss == 'crossentropy'):
            pred_probs_X = F.softmax(pred.X, dim=-1)
            pred_probs_E = F.softmax(pred.E, dim=-1)
            pred_probs_y = F.softmax(pred.y, dim=-1)
        else:
            pred_probs_X = torch.exp(pred.X)
            pred_probs_E = torch.exp(pred.E)
            pred_probs_y = torch.exp(pred.y)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, guidance=None):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask, guidance=guidance)

        # Normalize predictions
        if(self.cfg.guidance.loss == 'crossentropy'):
            probX0 = F.softmax(pred0.X, dim=-1)
            probE0 = F.softmax(pred0.E, dim=-1)
            proby0 = F.softmax(pred0.y, dim=-1)
        else:
            probX0 = torch.exp(pred0.X)
            probE0 = torch.exp(pred0.E)
            proby0 = torch.exp(pred0.y)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False, guidance = None):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask, guidance)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    #effectively returns p(G_0|G_t, y)
    def forward(self, noisy_data, extra_data, node_mask, guidance = None, train_step = False):
        bs = extra_data.X.size(0)

        #replicates the null guidance token for the whole batch size
        cf_null_token = self.cf_null_token.repeat((bs, 1))

        #if we don't pass a guidance vector, we use the null guidance token:
        if(guidance is None):
            guidance = cf_null_token
        elif(train_step and self.cfg.guidance.s > 1): #it has sense only if the guidance is not ENTIRELY equal to cf_null_token
            #proceeds to randomly mask the guidance
            #TODO: this likely will have to be fixed for text-based where
            #the null token is inplanted in the query itself (it's the "pad" token)
            #and hence all of this is already done implicitly

            #True = substitute with null token
            guidance_mask = torch.rand((bs, 1), device="cuda").to(guidance.device) < self.cfg.guidance.p_uncond

            #now it has the same size as cf_null_token
            guidance_mask = guidance_mask.repeat((1, cf_null_token.size(-1)))

            guidance = torch.where(guidance_mask, cf_null_token, guidance)

        #this is used during test when we sample n_samples_per_test_molecule
        #at once using the same guidance (=> the guidance has shape: (1, guidance_size)).
        #we need to repeat it n_samples_per_test_molecule (=> shape: (n_samples_per_test_molecule, guidance_size))
        #the "not train_step" is not necessary but better have it
        if(bs > guidance.size(0) and not train_step):
            guidance = guidance.repeat((bs, 1))

        #we put the production of X, E, y in a method as we need to 
        #recycle the whole code later on if we want to calculate
        #the same with the null token in place of the guidance.
        def produce_XEy(noisy_data, extra_data, guidance = None):
            #X      = transformer output which uses guidance (if any)
            #X_null = transformer output which uses the null token
            X = noisy_data['X_t'].clone().float()
            E = noisy_data['E_t'].clone().float()
            y = noisy_data['y_t'].clone().float()

            if(guidance != None):
                if(self.cfg.guidance.guidance_medium in ['y', 'both']):
                    g_y = guidance

                    #adds the guidance
                    y   = torch.hstack((y, g_y)).float()
                
                if(self.cfg.guidance.guidance_medium in ['XE', 'both']):
                    n  = extra_data.X.size(1) #number of nodes

                    #spreads the guidance on all noses
                    #g_X must have size (bs, n, features)
                    #g_E must have size (bs, n, n, features)
                    g_X = torch.reshape(guidance, shape = (bs,    1, -1)).repeat((1, n,    1))
                    g_E = torch.reshape(guidance, shape = (bs, 1, 1, -1)).repeat((1, n, n, 1))
                    
                    #adds the guidance
                    X   = torch.cat((X, g_X), dim=-1)
                    E   = torch.cat((E, g_E), dim=-1)

            #we finally add the extra data as requested
            X = torch.cat((X, extra_data.X), dim=2).float()
            E = torch.cat((E, extra_data.E), dim=3).float()
            y = torch.hstack((y, extra_data.y)).float()

            return X, E, y

        X, E, y = produce_XEy(noisy_data, extra_data, guidance)

        #p(x_0|x_t, guidance)
        out = self.model(X, E, y, node_mask)

        if(self.cfg.guidance.loss == 'crossentropy'):
            if(train_step == False and self.cfg.guidance.s > 1):
                #first of all, we need to calculate p(x_0|x_t, None) as well:
                X_null, E_null, y_null = produce_XEy(noisy_data, extra_data, cf_null_token)
                out_null = self.model(X_null, E_null, y_null, node_mask)

                out.X = out_null.X + self.cfg.guidance.s*(out.X - out_null.X)
                out.E = out_null.E + self.cfg.guidance.s*(out.E - out_null.E)
            
            #if the loss is a crossentropy, we are done here.
            #the raw outputs are the only thing we need.
            return utils.PlaceHolder(X = out.X, E = out.E, y = out.y).mask(node_mask)
        elif(self.cfg.guidance.loss in ['kl', 'nll']):
            #convert to log_softmax. Will be normalized later
            out.X = torch.log_softmax(out.X, dim=-1)
            out.E = torch.log_softmax(out.E, dim=-1)

            if(train_step or self.cfg.guidance.s <= 1):
                #s <= 1 means that we either do not want to use guidance at all
                #OR we want to use the ORIGINAL VQ loss. In either case, there
                #is no need to calculate the loss with the null token

                #we assume that the model outputs the UNNORMALIZED log probs.
                #Thus, we normalize them back.
                out_X = out.X - torch.logsumexp(out.X, dim=-1, keepdim=True)
                out_E = out.E - torch.logsumexp(out.E, dim=-1, keepdim=True)

                return utils.PlaceHolder(X = out_X, E = out_E, y = out.y).mask(node_mask)
            else:
                #first of all, we need to calculate p(x_0|x_t, None) as well:
                X_null, E_null, y_null = produce_XEy(noisy_data, extra_data, cf_null_token)
                out_null = self.model(X_null, E_null, y_null, node_mask)

                out_null.X = torch.log_softmax(out_null.X, dim=-1)
                out_null.E = torch.log_softmax(out_null.E, dim=-1)

                probX0 = out_null.X + self.cfg.guidance.s*(out.X - out_null.X)
                probX0 = probX0 - torch.logsumexp(probX0, dim=-1, keepdim=True)
                
                probE0 = out_null.E + self.cfg.guidance.s*(out.E - out_null.E)
                probE0 = probE0 - torch.logsumexp(probE0, dim=-1, keepdim=True)

                return utils.PlaceHolder(X = probX0, E = probE0, y = out.y).mask(node_mask)
        else:
            raise NotImplementedError("ERROR: unimplemented loss")

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, guidance=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        if(guidance == None):
            guidance = self.cf_null_token.repeat((batch_size, 1))

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, g_T = guidance)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y



        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end=''""", flush=True""")
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, g_T = None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, g_T)

        # Normalize predictions
        if(self.cfg.guidance.loss == 'crossentropy'):
            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        else:
            pred_X = torch.exp(pred.X)
            pred_E = torch.exp(pred.E)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
