# minLeWM

**🚧 Work in Progress 🚧**

A minimal PyTorch implementation of "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero [1]

minLeWM is an attempt to implement LeWorldModel from scratch as much as possible by only following the paper, as a fun exercise. 

## Model Architecture
minLeWM consists of two main components: an encoder and a predictor. 
- The encoder is a ViT that projects image frames into latent space. The final [CLS] token embedding is used as the latent embedding for that frame. 
- The predictor is a causal transformer that takes in a history of N embeddings and actions to predict the embedding of the next frame. AdaLN is used to inject the action information. 

In addition, a linear layer with batch normalisation is applied to the output embedding of the encoder because layer normalization in the final layer of ViT interferes with the optimisation of SIGReg. The same is done for the predictor. 

A decoder is also used to visualise latent states. It is a lightweight transformer that uses the latent embedding from the encoder as keys and values to perform cross attention with a fixed set of learnable query tokens for the target image. The paper only specifies the image and patch sizes of the decoder, we assume the rest of the configuration is the same as the encoder. 

## Training Objective
minLeWM uses a criterion with two terms: a prediction loss and a regularisation loss. 
- The prediction loss is the MSE between the predicted and actual latent embeddings.
- The regularisation loss is the SIGReg [2] function adopted by the paper.

## Dataset
Currently, only the PushT environment and dataset is supported. The LeWorldModel paper specifies that the setup follows that in DINO-WM [3]. The dataset can be found [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28).

## References

[[1](https://arxiv.org/abs/2603.19312)] Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. _LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels_ arXiv:2603.19312

[[2](https://rezabyt.github.io/blogposts/sigreg-tutorial.html)] Bayat, R. _SIGReg from First Principles_ 

[[3](https://github.com/gaoyuezhou/dino_wm/tree/main)] Zhou G., Pan, H., Lecun, Y., Pinto, L. _DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning_