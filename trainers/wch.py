import torch

from trainers.base import BaseTrainer


class WCHTrainer(BaseTrainer):

    def compute_features_one_batch(self, data, contrastive=True):
        device = self.device

        image, labels, index = data
        if contrastive:
            image_1, image_2 = image
            image_1, image_2 = image_1.to(device), image_2.to(device)

            data = (image_1, image_2), labels, index
            output = self.model.train_forward(image_1, image_2)
        else:
            image = image.to(device)
            data = image, labels, index
            output = self.model.forward(image)

        # print(output[1][0][:4].detach(), output[1][1][:4].detach())

        return data, self.parse_model_output(output)

    def parse_model_output(self, output):
        if isinstance(output, tuple):
            h1, h2, weighted = output
            return {
                'h1': h1,
                'h2': h2,
                'weighted': weighted
            }
        else:
            return {
                'codes': output
            }

    def inference_one_batch(self, *args, **kwargs):
        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data, contrastive=False)
            labels = data[1]

        return {
            'codes': output['codes'],
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        data, meters = args

        # clear gradient
        self.optimizer.zero_grad()

        data, output = self.compute_features_one_batch(data)

        h1, h2, weighted = output['h1'], output['h2'], output['weighted']
        loss = self.criterion(h1, h2, weighted)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), h1.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), h1.size(0))
