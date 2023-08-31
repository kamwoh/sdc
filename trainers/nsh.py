import torch

from trainers.base import BaseTrainer


class NSHTrainer(BaseTrainer):

    def compute_features_one_batch(self, data, contrastive=True):
        device = self.device

        image, labels, index = data
        if contrastive:
            image_1, image_2 = image
            image_1, image_2 = image_1.to(device), image_2.to(device)

            data = (image_1, image_2), labels, index
            concat_image = torch.cat([image_1, image_2], dim=0)
        else:
            image = image.to(device)
            data = image, labels, index
            concat_image = image

        output = self.model(concat_image)
        # print(output[1][0][:4].detach(), output[1][1][:4].detach())

        return data, self.parse_model_output(output)

    def parse_model_output(self, output):
        feats, codes, latents = output
        return {
            'feats': feats,
            'codes': codes,
            'latents': latents
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

        b = output['codes']
        z = output['latents']
        loss = self.criterion(b, z)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), b.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), b.size(0))
