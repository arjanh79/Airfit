import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

from PUSHUPS.pushups_trainer import PushUpsModel


class Workout:
    def __init__(self):
        self.model = PushUpsModel()
        self.datafile = 'PUSHUPS/progress.txt'
        self.model.load_state_dict(torch.load('PUSHUPS/best_model.pth', weights_only=True))
        self.workout_length = 11

    def optimize(self):

        base_reps = torch.tensor([[2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2]], dtype=torch.float32)
        multiplier = self.get_multiplier()
        reps = base_reps * multiplier

        start_reps = reps.tolist()
        start_reps = ' '.join([f'{i:.02f}' for i in start_reps[0]])
        print(f'START REPS: [{start_reps}]\n')

        reps.requires_grad = True

        target = torch.ones((1, self.workout_length))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW([reps], lr=0.05, weight_decay=0.03)
        loss_fn = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        best_reps = None
        no_improves = 0

        for i in range(5000):
            optimizer.zero_grad()
            predict = self.model(reps)

            loss = loss_fn(predict, target)

            if best_loss - loss.item() > 0.0001:
                best_loss = loss.item()
                best_reps = reps.detach().clone()
                no_improves = 0
                sign = "+"
            else:
                no_improves += 1
                sign = "-"
                if no_improves == 50:
                    break


            probs = predict.sigmoid().tolist()[0]
            probs = ' '.join([f'{i:.05f}' for i in probs])
            print(f'{i:04d} {loss.item():.05f} - [{probs}] => {sign}')

            if torch.min(predict) > 1.386:
                best_reps = reps
                break

            loss.backward()
            optimizer.step()


        reps = best_reps
        reps = (reps // 2 * 2).to(dtype=torch.int8)

        for _ in range(2):
            logits = self.model(reps)
            reps[:, torch.argmax(logits)] += 2  # Easiest exercise +2

        score = self.model(reps).sigmoid()

        print(f'\nWorkout = {reps[0].tolist()}')

        score = ' '.join([f'{i:.05f}' for i in score[0]])
        print(f'Score = [{score}]')
        return reps

    def get_multiplier(self):
        data = np.genfromtxt(self.datafile, delimiter=',')
        data = data[::-1].copy()  # copy to avoid issues with flipping
        x = torch.tensor(data[:,:self.workout_length], dtype=torch.float32)
        y = torch.tensor(data[:,self.workout_length:], dtype=torch.float32)

        x = x * y  # Filter out the fails.

        max_completed_reps = torch.max(x, dim=0)[0].unsqueeze(0)

        print(f'  MAX REPS: {max_completed_reps.to(torch.int8).tolist()[0]}')

        max_completed_reps[:, [0, -1]] = max_completed_reps[:, [0, -1]] / 2

        return torch.mean(max_completed_reps)


if __name__ == '__main__':
    wo = Workout()
    wo.optimize()