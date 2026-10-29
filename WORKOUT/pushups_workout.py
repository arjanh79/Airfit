import itertools
from datetime import datetime

import torch
import torch.optim as optim


from WORKOUT.pushups_trainer import PushUpsModel


class Workout:
    def __init__(self):
        self.model = PushUpsModel()
        self.datafile = 'WORKOUT/progress.txt'
        self.model.load_state_dict(torch.load('WORKOUT/best_model.pth', weights_only=True))
        self.workout_length = 11
        self.var_mask = torch.tensor([[2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2]])

        day_of_year = datetime.now().timetuple().tm_yday
        self.gen = torch.Generator().manual_seed(day_of_year)

    def optimize(self):

        reps = self.var_mask.to(torch.float)


        start_reps = reps.tolist()
        start_reps = ' '.join([f'{i:.02f}' for i in start_reps[0]])
        print(f'START REPS: [{start_reps}]\n')

        reps.requires_grad = True

        penalty_weight = 7

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW([reps], lr=0.05, weight_decay=0.001, betas=(0.85, 0.999))

        best_reps = reps.clone().detach()

        for i in range(1000):
            optimizer.zero_grad()
            predict = self.model(reps)

            p_success = predict.sigmoid()

            penalty = (1 - p_success)
            penalty_loss = ((penalty_weight * penalty) ** 0.5).mean()
            var_loss = torch.var(reps / self.var_mask, unbiased=True)

            reps_loss = -reps.mean()

            loss = reps_loss + var_loss + penalty_loss

            probs_score = predict.sigmoid()

            if torch.min(probs_score) < 0.9:
                break

            min_prob = f'{torch.min(probs_score).item():.05f}'
            probs_score = probs_score.tolist()[0]
            probs = ' '.join([f'{i:.05f}' for i in probs_score])
            print(f'{i:04d} {loss.item():.05f} - [{probs}] [{min_prob}]')

            best_reps = reps.clone().detach()

            loss.backward(retain_graph=True)

            optimizer.step()

            if not reps.grad is None:
                if torch.max(torch.abs(reps.grad)) < 0.01:
                    break

        reps = (best_reps // 2 * 2).to(dtype=torch.int8)

        reps_high = reps + 2

        linked = zip(reps.squeeze().tolist(), reps_high.squeeze().tolist())
        combined = list(itertools.product(*linked))
        combined = torch.tensor(combined)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(combined).sigmoid()
            pred_mask = torch.sum(predictions > 0.9, dim=1) == 11
            combined = combined[pred_mask]

            total_reps = torch.sum(combined, dim=1)
            max_reps = torch.max(total_reps)
            combined = combined[torch.where(total_reps == max_reps)]

            total_variance = torch.var(combined / self.var_mask, unbiased=True, dim=1)
            min_variance = torch.min(total_variance)
            combined = combined[(total_variance == min_variance)]

            total_variance = torch.var(self.model(combined).sigmoid(), unbiased=True, dim=1)
            min_variance = torch.min(total_variance)
            best_workout = torch.where(total_variance == min_variance)[-1]

            reps = combined[best_workout]

        score = self.model(reps).sigmoid()

        scores = list(zip(reps, score, best_reps))

        print(' Reps\t\t Scores\t\t  Model')
        for idx in range(len(scores[0][0])):
            print(f'\t{scores[0][0][idx].item()}\t\t{scores[0][1][idx].item():.05f}\t\t{scores[0][2][idx].item():.05f}')


        return reps


if __name__ == '__main__':
    wo = Workout()
    wo.optimize()