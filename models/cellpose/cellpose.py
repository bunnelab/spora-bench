from spora_bench.wrapper import SporaModelWrapper




class SporaCellposeWrapper(SporaModelWrapper):

    def __init__(self,
                 model_name: str,
                 ):
        super().__init__(model_name)
        raise NotImplementedError("SporaCellposeWrapper is not implemented yet. Please implement the necessary methods for this wrapper.")