from easydict import EasyDict
import tensorflow as tf

import mynetwork

from blueoil.common import Tasks
from blueoil.datasets.image_folder import ImageFolderBase
from blueoil.data_augmentor import (
    Blur,
    Brightness,
    Color,
    FlipLeftRight,
    SSDRandomCrop,
)
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    DivideBy255,
    PerImageStandardization
)
from blueoil.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = mynetwork.MyNetworkQuantize

# TODO(wakisaka): should be hidden. generate dataset class on the fly.
DATASET_CLASS = type('DATASET_CLASS', (ImageFolderBase,), {'extend_dir': '/storage/iizuka/cifar/train', 'validation_extend_dir': '/storage/iizuka/cifar/test'})

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 64
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

MAX_EPOCHS = 100
SAVE_CHECKPOINT_STEPS = 1000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 100


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255()
])
POST_PROCESSOR = None

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {'momentum': 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.train.cosine_decay
NETWORK.LEARNING_RATE_KWARGS = {'learning_rate': 0.1, 'decay_steps': 78125}

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005

# quantize
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Blur(value=(0, 1), ),
    Brightness(value=(0.75, 1.25), ),
    Color(value=(0.75, 1.25), ),
    FlipLeftRight(probability=0.5, ),
])
DATASET.ENABLE_PREFETCH = True
