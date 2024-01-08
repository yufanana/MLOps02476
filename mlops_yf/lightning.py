from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

dataset = WikiText2()
dataloader = DataLoader(dataset)
model = LightningTransformer(vocab_size=dataset.vocab_size)

trainer = L.Trainer(fast_dev_run=100)
trainer.fit(model=model, train_dataloaders=dataloader)