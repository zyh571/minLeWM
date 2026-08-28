import torch
from pathlib import Path

def train_epoch(train_loader, model, optimizer, log_batch_interval=10, device='cuda'):
    model.train()
    train_loss = {'loss': 0, 'pred_loss': 0, 'sigreg_loss': 0}
    batch_losses = []
    for idx, batch in enumerate(train_loader):
        obs = batch[0]['visual'].to(device)
        action = batch[1].to(device)
        loss_dict = model(obs, action)
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        train_loss = {k: train_loss[k] + v.item() for k, v in loss_dict.items()}
        batch_losses.append({k: v.item() for k, v in loss_dict.items()})
        if idx % log_batch_interval == 0:
            loss, pred_loss, sigreg_loss = [v.item() for v in loss_dict.values()]
            print(f"    Batch {idx}, loss: {loss:.5f}, pred_loss: {pred_loss:.5f}, sigreg_loss: {sigreg_loss:.5f}")

    train_loss = {k: v/len(train_loader) for k, v in train_loss.items()}
    return train_loss, batch_losses

@torch.no_grad()
def evaluate(valid_loader, model, device='cuda'):
    model.eval()
    valid_loss = {'loss': 0, 'pred_loss': 0, 'sigreg_loss': 0}
    for batch in valid_loader:
        obs = batch[0]['visual'].to(device)
        action = batch[1].to(device)
        loss_dict = model(obs, action)
        valid_loss = {k: valid_loss[k] + v.item() for k, v in loss_dict.items()}

    valid_loss = {k: v/len(valid_loader) for k, v in valid_loss.items()}
    return valid_loss

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu') # loads on cpu in case no gpu
    epoch_start = checkpoint['epoch'] + 1
    epoch_end = checkpoint['epoch_end']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    batch_losses = checkpoint['batch_losses']        
    print(f"Resumed from {path}")
    return epoch_start, epoch_end, train_losses, valid_losses, batch_losses

def save_checkpoint(path, epoch, epoch_end, model, optimizer, train_losses, valid_losses, batch_losses):
    checkpoint = {
        'epoch': epoch,
        'epoch_end': epoch_end,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'batch_losses': batch_losses,
    }
    torch.save(checkpoint, path)
    print(f"Saved to {path}")

def train(epochs, train_loader, valid_loader, model, optimizer,
          resume_path=None, log_batch_interval=10,
          save_dir='./runs', save_name='checkpoint', save_epoch_interval=1,
          device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    epoch_start = 0
    epoch_end = epochs
    train_losses = [] # per epoch loss dicts
    valid_losses = [] # per epoch loss dicts
    batch_losses = [] # per batch loss dicts
    
    # resume from checkpoint, will override epochs
    if resume_path is not None:
        epoch_start, epoch_end, train_losses, valid_losses, batch_losses = load_checkpoint(resume_path, model, optimizer)
    model.to(device)
    for state in optimizer.state.values(): # make sure optimizer is on same device as model
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    # ensure save_dir is valid
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epoch_start, epoch_end):
        print(f"Epoch {epoch+1} Training...\n")
        train_loss, epoch_batch_losses = train_epoch(train_loader, model, optimizer, log_batch_interval, device)
        valid_loss = evaluate(valid_loader, model, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        batch_losses.extend(epoch_batch_losses)

        # only prints total loss because im lazy
        print(f"\nEpoch {epoch+1} Completed.")
        print(f"Train Loss: {train_loss['loss']:.5f} | Valid Loss: {valid_loss['loss']:.5f}\n")

        # save checkpoint
        if ((epoch+1) % save_epoch_interval == 0) or (epoch == epoch_end-1):
            save_path = f"{save_dir}/{save_name}_{epoch}.pt"                
            save_checkpoint(save_path, epoch, epoch_end, model, optimizer, train_losses, valid_losses, batch_losses)

    return train_losses, valid_losses, batch_losses