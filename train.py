import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model.lewm import LeWorldModel
from criterion import LeWMLoss
from dataset.pusht_dset import load_pusht_slice_train_val

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.Resize((224, 224))])
dset, traj_dset = load_pusht_slice_train_val(
    transform=None,    
    n_rollout=None,
    data_path='dataset/data/pusht_noise',
    normalize_action=True,
    split_ratio=0.8,
    num_hist=3,
    num_pred=1,
    frameskip=5,
    with_velocity=True,
)

train_dset = dset['train']
valid_dset = dset['valid']

train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
model = LeWorldModel().to(device)
loss_fn = LeWMLoss()
opt = torch.optim.AdamW(params=model.parameters())

for epoch in range(10):
    for batch in train_loader:
        obs = batch[0]['visual'].to(device)
        action = batch[1].to(device)
        z_pred, z_targ, z = model(obs, action)
        loss = loss_fn(z_pred, z_targ, z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
