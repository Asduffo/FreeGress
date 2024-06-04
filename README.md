# Classifier-free graph diffusion for molecular property targeting

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```
  - `conda activate digress`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```


## Testing DiGress/FreeGress on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. Depending on whether you are considering
molecules or abstract graphs, you can base this file on `moses_dataset.py` or `spectre_datasets.py`, for example. 
This file should implement a `Dataset` class to process the data (check [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)), 
as well as a `DatasetInfos` class that is used to define the noise model and some metrics.

For molecular datasets, you'll need to specify several things in the DatasetInfos:
  - The atom_encoder, which defines the one-hot encoding of the atom types in your dataset
  - The atom_decoder, which is simply the inverse mapping of the atom encoder
  - The atomic weight for each atom atype
  - The most common valency for each atom type

The node counts and the distribution of node types and edge types can be computed automatically using functions from `AbstractDataModule`.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.


## Running the code (FreeGress)
For a generic training on the QM9 dataset, using p_uncond=0.1, HOMO as the target:

```python3 main.py +experiment=guidance_parameters.yaml general.name='homo_cf_p01_noextra' guidance.guidance_target=["homo"] model.extra_features=null guidance.n_test_molecules_to_sample=4 guidance.n_samples_per_test_molecule=5 train.ckpt_every_n_train_steps=25 general.check_val_every_n_epochs=25 general.gpus=[0] guidance.p_uncond=0.1 train.num_interops_workers=1 general.sample_every_val=9999 general.log_every_steps=191;```

The general idea is the following:
 - ```general.name``` is the name of the model
 - ```guidance.guidance_target``` should be a list of all the target properties (for instance, if we were to use both homo and mu instead, we would have ```guidance.guidance_target=["mu","homo"]```
 - ```guidance.n_test_molecules_to_sample``` and ```guidance.n_samples_per_test_molecule```: leave them to low values for the time being. They are used only during a proper test
 - ```train.ckpt_every_n_train_steps```: the number of training epochs between the various checkpoints saved. Should be set equal to general.check_val_every_n_epochs
 - ```general.gpus```: GPU ID. If you want to train using multiple GPUs, pass a list of IDs (es: general.gpus=[0,1])
 - ```guidance.p_uncond```: the p_uncond hyperparameter (rho in the paper)
 - ```train.num_interops_workers```: leave it to one
 - ```general.sample_every_val```: leave it to 9999
 - ```model.extra_features```: null if you do not want to use the extra features. 'domain_only' to use only molecular features, 'all' to use spectral features as well
 - ```general.log_every_steps```: this parameter is a bit complicated because of how wandb works, but it should be equal to the number of batches in the training set (dataset_size / batch_size, to be clear). Don't pay too much attention to it since it is used only for logging anyway.

Note that the tests launched at the end of the training should not be used for any comparative result as some machines may randomly change the RNG during the test phase. It is much easier to just launch the proper test separately. To test a classifier-free model, check the ```/FreeGress/output folder```. Search for the folder named after the date when you trained the model in the format ```yyyy-mm-dd```. Inside of it, there should be a folder named ```<hour>-<general.name>```, where ```<hour>``` is the time when you started to train the model, in the format ```hh-mm-ss```. The checkpoints are stored inside the folder ```checkpoints/<general.name>/```. To launch a model, execute the command

```python3 main.py +experiment=guidance_parameters.yaml general.name='homo_cf_p01_noextra_s5' guidance.guidance_target='homo' model.extra_features=null guidance.n_test_molecules_to_sample=100 guidance.n_samples_per_test_molecule=10 train.batch_size=1 general.gpus=[0] train.num_interops_workers=1 general.sample_every_val=9999 general.log_every_steps=191 guidance.guidance_target=["homo"] guidance.s=5 general.test_only="FULL CHECKPOINT PATH"```

 -```guidance.s``` is the s hyperparameter

Classifier-based models are slightly different. The unconditional part is trained using something on the line of

```python3 train_unconditional.py +experiment=guidance_parameters.yaml general.name="cb_qm9" general.gpus=[0] general.log_every_steps=191 model.extra_features=null guidance.guidance_target=NONE  guidance.guidance_medium="NONE"```

 - The combination of guidance.guidance_medium="NONE" and guidance.guidance_target=NONE says that there is no guidance vector

To train the regressor, use something on the line of 

```python3 guidance/train_qm9_regressor.py +experiment=guidance_parameters.yaml general.name="cb_qm9_homo_regressor" general.gpus=[0] model.extra_features=null train.n_epochs=1000 train.ckpt_every_n_train_steps=5 general.check_val_every_n_epochs=5 guidance.guidance_target=["homo"] guidance.guidance_medium="NONE"```

 - ```train.n_epochs=1000``` may be reduced. From my personal experience with zinc, it tends to overfit after 100-200 epochs anyway
 - I suggest keeping ```train.ckpt_every_n_train_steps``` and ```general.check_val_every_n_epochs``` relatively low since the validation accuracy tends to swing around easily.

To test a classifier-based model, the command is the same as above (just with guidance.guidance_medium="NONE", which tells the model to don't attach the guidance vector at the end of the inputs).
```python3 guidance/main_guidance.py +experiment=guidance_parameters.yaml general.wandb="online" general.name="cb_qm9_homo_test_100 guidance.guidance_target=["homo"] model.extra_features=null  +general.test_samples_to_generate=100 +general.test_only="PATH TO UNCONDITIONAL CHECKPOINT" general.gpus=[0] guidance.guidance_medium="NONE" train.batch_size=1 +general.trained_regressor_path="PATH TO REGRESSOR" +guidance.lambda_guidance=100;```

 - ```guidance.lambda_guidance``` is the \lambda hyperparameter

To train on ZINC-250k instead, add
 - ```+experiment=zinc250k.yaml```
 - ```dataset=zinc250k```
