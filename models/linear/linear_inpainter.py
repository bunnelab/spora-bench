import torch
from spora_bench.wrapper import SporaModelWrapper
from spora_io.datasets._types import MultiplexTissue

import joblib


class LinearRegressionInpainter(SporaModelWrapper):

    def __init__(self, 
                 model_name: str,
                 checkpoint_path: str
                ):
        super().__init__(model_name)
        self.regression_models = joblib.load(checkpoint_path)


    def predict_marker(self,
                tissue: MultiplexTissue,
                target_channel_name: str,
                target_uniprot_id: str,
                ):
        regression_model = self.regression_models[target_channel_name]

        x = tissue.image # (C,H,W)
        x = x.numpy()
        H, W = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], -1).T # (H*W, C)

        y_pred = regression_model.predict(x) # (H*W,)
        y_pred = y_pred.reshape(H, W) # (H, W)

        predicted_marker = torch.from_numpy(y_pred).unsqueeze(0) # (1, H, W)
        return predicted_marker
        

