import torch
from torch.utils.data import Dataset

import numpy as np


class PushUpsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.workout_length = 11
        self.datafile = 'progress.txt'
        self.x, self.y = self.read_data()
        self.weighted_loss = self.create_weighted_loss()
        # self.create_baseline()


    def read_data(self):
        data = np.genfromtxt(self.datafile, delimiter=',')
        data = data[::-1].copy()  # copy to avoid issues with flipping
        x = torch.tensor(data[:,:self.workout_length], dtype=torch.int64)
        y = torch.tensor(data[:,self.workout_length:], dtype=torch.float32)

        return x, y


    def create_baseline(self):
        y_base = torch.cat((torch.zeros((1, self.workout_length)), torch.ones((1, self.workout_length))), 0)
        x_base = torch.zeros_like(y_base)

        x_base[0, :] = 20
        x_base[0, [0, -1]] = 40

        x_base[1, :] = 3
        x_base[1, [0, -1]] = 10

        self.x = torch.cat((x_base, self.x), dim=0)
        self.y = torch.cat((y_base, self.y), dim=0)
        self.weighted_loss = torch.cat((torch.ones(2, 1), self.weighted_loss))



    def create_weighted_loss(self):
        num_workouts = len(self.y)
        weighted_loss = torch.linspace(0, torch.pi, 20)
        weighted_loss = (torch.cos(weighted_loss) + 1.1) / 2.1
        weighted_loss = torch.clamp(weighted_loss, 0.05, 1)

        if len(weighted_loss) < num_workouts:
            to_add = num_workouts - len(weighted_loss)
            added_loss = torch.zeros(to_add) + 0.05
            weighted_loss = torch.cat((weighted_loss, added_loss))


        return weighted_loss[:num_workouts].unsqueeze(-1)




    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.weighted_loss[index]



if __name__ == '__main__':
    xxx = PushUpsDataset()