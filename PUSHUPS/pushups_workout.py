from datetime import datetime

import torch
import torch.optim as optim


from PUSHUPS.pushups_trainer import PushUpsModel


class Workout:
    def __init__(self):
        self.model = PushUpsModel()
        self.datafile = 'PUSHUPS/progress.txt'
        self.model.load_state_dict(torch.load('PUSHUPS/best_model.pth', weights_only=True))
        self.workout_length = 11

        day_of_year = datetime.now().timetuple().tm_yday
        self.gen = torch.Generator().manual_seed(day_of_year)

    def optimize(self):

        base_reps = torch.tensor([[3, 2, 2, 3, 2, 3, 2, 2, 3, 2, 3]], dtype=torch.float32)

        modifier = torch.where(torch.rand(11, generator=self.gen) < 0.5, -1, 1).to(torch.float32)

        reps = base_reps + modifier

        start_reps = reps.tolist()
        start_reps = ' '.join([f'{i:.02f}' for i in start_reps[0]])
        print(f'START REPS: [{start_reps}]\n')

        reps.requires_grad = True

        penalty_weight = 3

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW([reps], lr=0.05, weight_decay=0.001)

        best_reps = reps.clone().detach()

        for i in range(1000):
            optimizer.zero_grad()
            predict = self.model(reps)

            p_success = predict.sigmoid()

            penalty = (0.8 - p_success)
            penalty = torch.where(penalty < 0, penalty * 0.1, penalty)

            loss = -reps + penalty_weight * penalty
            loss = torch.mean(loss)

            probs_score = predict.sigmoid().tolist()[0]
            probs = ' '.join([f'{i:.05f}' for i in probs_score])

            print(f'{i:04d} {loss.item():.05f} - [{probs}]')

            if loss < -9.5:
                break

            best_reps = reps.clone().detach()

            loss.backward()
            optimizer.step()


        reps = best_reps + 1
        reps = (reps // 2 * 2).to(dtype=torch.int8)

        score = self.model(reps).sigmoid()

        print(f'Workout = {reps[0].tolist()}')

        score = ' '.join([f'{i:.05f}' for i in score[0]])
        print(f'  Score = [{score}]\n')
        return reps



if __name__ == '__main__':
    wo = Workout()
    wo.optimize()