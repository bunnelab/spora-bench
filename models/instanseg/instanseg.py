from spora_bench.wrapper import SporaModelWrapper
from instanseg.inference_class import InstanSeg

class SporaInstansegWrapper(SporaModelWrapper):

    def __init__(self,
                 model_name: str,
                 ):
        super().__init__(model_name)
        self.model = InstanSeg(
            model_type="fluorescence_nuclei_and_cells",
        )

    def predict_instance_segmentation(self, tissue):
        """
        Predicts instance segmentation for the given tissue using the InstanSeg model.
        Args:
            tissue: The input tissue to be segmented.
        Returns:
            predicted_mask: The predicted instance segmentation mask.
        """
        predicted_mask = self.model.eval_medium_image(
            tissue.image,
            pixel_size=1.0,
            normalise=True,
            return_image_tensor=False,
            target="cells",
        )
        return predicted_mask[0,0].int()
