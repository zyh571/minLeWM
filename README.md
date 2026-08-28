# minLeWM

**🚧 Work in Progress 🚧**

A minimal PyTorch implementation of "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero [[1](#references)]

minLeWM is an attempt to implement LeWorldModel from scratch by only following the paper to the best of my interpretation as a fun exercise. 

## Status
- [x] Setup: environment, dataset
- [x] Model architecture: encoder, predictor, decoder
- [x] Loss function: MSE loss, SIGReg regularisation
- [x] Latent planning: Cross Entropy Method, Model Predictive Control
- [x] Training Loop: logging, checkpointing
- [ ] Experiment: full training runs, evaluation

## Setup
**1. Install**
```bash
git clone https://github.com/zyh571/minLeWM && cd minLeWM
uv sync
```
**2. Dataset** 

Currently, only the PushT environment and dataset is supported. Download the [Push-T dataset](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) from DINO-WM [[2](#references)] and extract to `dataset/data/pusht_noise/`.

**3. Train**
```python
import torch
from torch.utils.data import DataLoader
from dataset.pusht_dset import load_pusht_slice_train_val
from model.lewm import LeWorldModel
from train import train

dset, traj_dset = load_pusht_slice_train_val(
    n_rollout=None,
    data_path='dataset/data/pusht_noise',
    num_hist=3,
    num_pred=1,
    frameskip=5,
)

train_dset = dset['train']
valid_dset = dset['valid']
train_loader = DataLoader(train_dset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dset, batch_size=128, shuffle=False)

model = LeWorldModel()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

train_losses, valid_losses, batch_losses = train(
    epochs=10,
    train_loader=train_loader,
    valid_loader=valid_loader, 
    model=model, 
    optimizer=optimizer,
)
```

For more code examples, refer to `playground.ipynb`. 
## Model Architecture
minLeWM consists of two main components: an encoder and a predictor. 
- The encoder is a ViT that projects image frames into latent space. The final [CLS] token embedding is used as the latent embedding for that frame. 
- The predictor is a causal transformer that takes in a history of N embeddings and actions to predict the embedding of the next frame. AdaLN is used to inject the action information. 

In addition, a linear layer with batch normalisation is applied to the output embedding of the encoder because layer normalization in the final layer of ViT interferes with the optimisation of SIGReg. The same is done for the predictor. 

A decoder is also used to visualise latent states. It is a lightweight transformer that uses the latent embedding from the encoder as keys and values to perform cross attention with a fixed set of learnable query tokens for the target image. The paper only specifies the image and patch sizes of the decoder, we assume the rest of the configuration is the same as the encoder. 

## Training Objective
minLeWM uses a criterion with two terms: a prediction loss and a regularisation loss. 
- The prediction loss is the MSE between the predicted and actual latent embeddings.
- The regularisation loss is the SIGReg function adopted in the paper. See [[3](#references)] for a helpful implementation guide.

## Ambiguities
**Predictor initial state.** §3.2 writes `ẑ_1 = enc(o_1)`, implying a single frame, but §3.1 states the predictor consumes a history of N frames and App. D sets N = 3 for Push-T. minLeWM seeds the rollout with 3 encoded frames.

 **Action indexing in the planner.** Eq. 5 optimises over `a_{1:H}`, but the rollout `ẑ_{t+1} = pred(ẑ_t, a_t)` initialised at `ẑ_1` reaches `ẑ_H` after `H−1` transitions. `a_H` never enters the terminal cost, so it is an unidentifiable dummy variable. minLeWM optimises `a_{1:H−1}`.

## References

[[1](https://arxiv.org/abs/2603.19312)] Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. _LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels_ arXiv:2603.19312

[[2](https://github.com/gaoyuezhou/dino_wm/tree/main)] Zhou G., Pan, H., Lecun, Y., Pinto, L. _DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning_

[[3](https://rezabyt.github.io/blogposts/sigreg-tutorial.html)] Bayat, R. _SIGReg from First Principles_ 

