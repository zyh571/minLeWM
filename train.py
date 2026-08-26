import torch
from torch.utils.data import DataLoader

from model.lewm import LeWorldModel
from criterion import LeWMLoss

# TBD: checkpointing and resuming
# TBD: add scheduler?
def train_model(dset, epochs=10, batch_size=128, log_interval=10,
                checkpoint=None, resume=False, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_dset = dset['train']
    valid_dset = dset['valid']

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=batch_size, shuffle=False)

    model = LeWorldModel().to(device)
    loss_fn = LeWMLoss()
    opt = torch.optim.AdamW(params=model.parameters())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for idx, batch in enumerate(train_loader):
            obs = batch[0]['visual'].to(device)
            action = batch[1].to(device)
            z_pred, z_targ, z = model(obs, action)
            loss = loss_fn(z_pred, z_targ, z)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.detach()
            if idx % log_interval == 0:
                print(f"    Batch {idx} Loss: {loss.item():.5f}")

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in valid_loader:
                obs = batch[0]['visual'].to(device)
                action = batch[1].to(device)
                z_pred, z_targ, z = model(obs, action)
                loss = loss_fn(z_pred, z_targ, z)
                val_loss += loss.detach()
            val_loss /= len(valid_loader)

        print(f"Epoch {epoch} | Train Loss: {train_loss.item():.5f} | Valid Loss: {val_loss.item():.5f}")

    return model