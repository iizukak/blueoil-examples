from simple_classification import cifar10
from blueoil.networks.classification.lmnet_v1 import LmnetV1Quantize


def test_instanciate_network():
    network_kwargs = {key.lower(): val for key, val in cifar10.NETWORK.items()}
    model = LmnetV1Quantize(
        classes=cifar10.CLASSES,
        is_debug=cifar10.IS_DEBUG,
        **network_kwargs,
    ) 
    assert len(model.classes) == 10
