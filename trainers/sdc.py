import torch

from trainers.base import BaseTrainer


class SDCTrainer(BaseTrainer):
    def forward_one_batch(self, images):
        outputs = self.model(images)
        if len(outputs) == 2:
            feats, codes = outputs
            return {
                'feats': feats,
                'codes': codes
            }
        else:
            feats, codes, cont_feats = outputs
            return {
                'feats': feats,
                'codes': codes,
                'cont_feats': cont_feats
            }

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        if isinstance(image, (tuple, list)):
            image_1, image_2 = image
            image = torch.cat([image_1, image_2], dim=0)
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            outputs = self.model(image)
            if len(outputs) == 2:
                feats, codes = outputs
                loss = self.criterion(feats, codes)
            else:
                feats, codes, cont_feats = outputs
                loss = self.criterion(feats, codes, cont_feats)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        if isinstance(image, (tuple, list)):
            image_1, image_2 = image
            image = torch.cat([image_1, image_2], dim=0)
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        outputs = self.model(image)
        if len(outputs) == 2:
            feats, codes = outputs
            loss = self.criterion(feats, codes)
        else:
            feats, codes, cont_feats = outputs
            loss = self.criterion(feats, codes, cont_feats)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
