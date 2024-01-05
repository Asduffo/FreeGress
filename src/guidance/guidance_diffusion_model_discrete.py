import numpy as np
import torch
import pytorch_lightning as pl
import time
import wandb
import os
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
import networkx as nx
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
import src.utils as utils
from src.metrics.properties import mw, penalized_logp, qed

# packages for conditional generation with guidance
from torchmetrics import MeanSquaredError, MeanAbsoluteError
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

from datasets import qm9_dataset, zinc_250k

class DiscreteDenoisingDiffusionUnconditional(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, guidance_model=None, load_model=False):
        super().__init__()

        # add for test
        if load_model:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.guidance = {'use_guidance': True, 'lambda_guidance': 0.5}

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
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

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
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
        # Marginal noise schedule
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

        # specific properties to generate molecules
        self.cond_val = MeanAbsoluteError()
        self.num_valid_molecules = 0
        self.num_total = 0

        self.guidance_model = guidance_model

        #stores the generated smiles on the test step
        self.generated_smiles = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, data, i):
        print(f'Select No.{i+1} test molecule')
        # Extract properties
        target_properties = data.guidance.clone()

        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)
        print("TARGET PROPERTIES", target_properties)

        start = time.time()

        ident = 0
        samples = self.sample_batch(batch_id=ident, batch_size=10, num_nodes=None,
                                    save_final=10,
                                    keep_chain=1,
                                    number_chain_steps=self.number_chain_steps,
                                    input_properties=target_properties)
        print(f'Sampling took {time.time() - start:.2f} seconds\n')

        self.save_cond_samples(samples, target_properties, file_path=os.path.join(os.getcwd(), f'cond_smiles{i}.pkl'))
        # save conditional generated samples
        mae = self.cond_sample_metric(samples, target_properties)
        return {'mae': mae}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
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
        elif(self.cfg.dataset.name == 'zinc250k'):
            train_dataset_smiles = zinc_250k.get_train_smiles(cfg=self.cfg, train_dataloader=None,
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

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, input_properties=None):
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
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, input_properties)
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
            print('Visualizing chains starts!')
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
                print('\r{}/{} complete'.format(i+1, num_molecules), end=''""", flush=True""")
            print('\nVisualizing chains Ends!')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)

        return molecule_list


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




    def cond_fn(self, noisy_data, node_mask, target=None):
        #self.guidance_model.eval()
        loss = nn.MSELoss()

        t = noisy_data['t']

        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['t']

        with torch.enable_grad():
            x_in = X.float().detach().requires_grad_(True)
            e_in = E.float().detach().requires_grad_(True)

            pred = self.guidance_model.model(x_in, e_in, y, node_mask)

            # normalize target
            target = target.type_as(x_in)

            mse = loss(pred.y, target.repeat(pred.y.size(0), 1))

            t_int = int(t[0].item() * 500)
            if t_int % 10 == 0:
                print(f'Regressor MSE at step {t_int}: {mse.item()}')
            wandb.log({'Guidance MSE': mse})

            # calculate gradient of mse with respect to x and e
            grad_x = torch.autograd.grad(mse, x_in, retain_graph=True)[0]
            grad_e = torch.autograd.grad(mse, e_in)[0]

            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            bs, n = x_mask.shape[0], x_mask.shape[1]

            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
            diag_mask = torch.eye(n)
            diag_mask = ~diag_mask.type_as(e_mask1).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

            mask_grad_x = grad_x * x_mask
            mask_grad_e = grad_e * e_mask1 * e_mask2 * diag_mask

            mask_grad_e = 1 / 2 * (mask_grad_e + torch.transpose(mask_grad_e, 1, 2))
            return mask_grad_x, mask_grad_e

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, input_properties):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
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
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

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

        # # Guidance
        lamb = self.cfg.guidance.lambda_guidance

        grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)

        p_eta_x = torch.softmax(- lamb * grad_x, dim=-1)
        p_eta_e = torch.softmax(- lamb * grad_e, dim=-1)

        prob_X_unnormalized = p_eta_x * prob_X
        prob_X_unnormalized[torch.sum(prob_X_unnormalized, dim=-1) == 0] = 1e-7
        prob_X = prob_X_unnormalized / torch.sum(prob_X_unnormalized, dim=-1, keepdim=True)

        prob_E_unnormalized = p_eta_e * prob_E
        prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
        prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)

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

    def save_cond_samples(self, samples, target, file_path):
        cond_results = {'smiles': [], 'input_targets': target}
        invalid = 0
        disconnected = 0

        print("\tConverting conditionally generated molecules to SMILES ...")
        for sample in samples:
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                cond_results['smiles'].append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        print("Number of invalid molecules", invalid)
        print("Number of disconnected molecules", disconnected)

        # save samples
        with open(file_path, 'wb') as f:
            pickle.dump(cond_results, f)

        return cond_results
