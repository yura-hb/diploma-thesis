from functools import partial

from utils import from_cli
from .dynamic import CLI as DynamicJobSamplerFromCLI, Builder as DynamicJobSamplerBuilder, \
    JobSampler as DynamicJobSampler
from .static import Sampler as StaticJobSampler

key_to_class = {
    "dynamic": DynamicJobSamplerFromCLI
}

from_cli = partial(from_cli, key_to_class=key_to_class)
