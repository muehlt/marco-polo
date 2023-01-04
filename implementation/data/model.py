from torch import nn
import torch

device = "cpu"


class RelevanceModel(nn.Module):
    # adapted code from torch quickstart
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = RelevanceModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()


def train(dataloader, model, loss_fn, optimizer):
    # adapted code from torch quickstart
    model.train()
    total_loss = 0
    for batch, (d, q, r) in enumerate(dataloader):

        # Compute prediction error
        pred = model((torch.tensor(d + q)).float())
        loss = loss_fn(pred, r[0].float())  # get tensor out of list
        total_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output loss every 1000 batches
        if batch % 1000 == 0:
            print(f"loss: {loss:>7f}")

    return total_loss

# TODO: implement test loop to predict relevance in test dataset

# Run model
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    # train(train_dataloader, model, loss_fn, optimizer) # TODO: use created dataloader
print("Done!")
