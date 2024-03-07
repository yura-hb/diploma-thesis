from functools import partial

from utils import from_cli
from .job_sampler import JobSampler
from .dynamic import CLI as DynamicJobSamplerFromCLI, Builder as DynamicJobSamplerBuilder, \
    JobSampler as DynamicJobSampler
from .static import No

key_to_class = {
    'no': No,
    "dynamic": DynamicJobSamplerFromCLI
}

from_cli = partial(from_cli, key_to_class=key_to_class)
