

import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset


from PUSHUPS.pushups_dataset import PushUpsDataset


class PushUpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.workout_length = 11

        embedding_dim = 4
        self.seq_emb = nn.Embedding(self.workout_length + 1, embedding_dim, padding_idx=0, max_norm=1)
        self.eid_emb = nn.Embedding(5 + 1, embedding_dim, padding_idx=0, max_norm=2)

        self.rep_block = nn.Sequential(nn.Linear(1, embedding_dim),
                                  nn.LayerNorm(embedding_dim),
                                  nn.ReLU(),
                                  nn.Linear(embedding_dim, embedding_dim)
                                )
        d_model = embedding_dim * 3

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):


        x_seq = torch.arange(1, self.workout_length+1).repeat(x.shape[0], 1)
        x_seq = self.seq_emb(x_seq)

        x_rep = x.float().unsqueeze(-1)
        x_rep = self.rep_block(x_rep)

        x_eid = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1], dtype=torch.int64).repeat(x.shape[0], 1)
        x_eid = self.eid_emb(x_eid)

        x = torch.cat((x_seq, x_rep, x_eid), dim=-1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1),
        )

        x = self.transformer(x, mask=causal_mask)

        logits = self.classifier(x).squeeze(-1)

        return logits




class PushUpsTrainer:
    def __init__(self):
        self.model = PushUpsModel()
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth', weights_only=True))

        self.ds = PushUpsDataset()
        test_samples = min(len(self.ds), 12)

        self.train_dl = DataLoader(self.ds, shuffle=False, batch_size=32)
        self.test_dl = DataLoader(Subset(self.ds, range(test_samples)), shuffle=False, batch_size=12)

        self.optimizer = optim.AdamW(self.model.parameters(), weight_decay=0.03)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')


    def train_one_epoch(self):
        self.model.train()
        for x, y, weighted_loss in self.train_dl:
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss = loss * weighted_loss

            loss = torch.mean(loss)
            loss.backward()
            self.optimizer.step()
        return self.eval()


    def eval(self):
        self.model.eval()
        with torch.no_grad():
            x, y, weighted_loss = next(iter(self.test_dl))
            loss = self.loss_fn(self.model(x), y)
            loss = loss * weighted_loss

        return loss.mean()



    def train_model(self):
        num_epochs = 10000
        best_loss = float('inf')
        no_improvement = 0
        for epoch in range(num_epochs):
            loss = self.train_one_epoch()
            print(f'{epoch:03d} {loss.item():.5f}')
            if (best_loss - loss) > 0.0001:
                best_loss = loss
                no_improvement = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('^^ BEST MODEL')
            else:
                no_improvement += 1
                if no_improvement == 50:
                    break



if __name__ == '__main__':
    pushups_trainer = PushUpsTrainer()
    pushups_trainer.train_model()
