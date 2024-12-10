import torch.optim as optim

def get_optimizer_and_scheduler(model, lr, step_size, gamma):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler
