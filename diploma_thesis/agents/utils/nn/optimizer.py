
import torch

from dataclasses import dataclass


@dataclass
class Configuration:
    @dataclass
    class Description:
        kind: str
        parameters: dict

        @staticmethod
        def from_cli(parameters):
            return Configuration.Description(
                kind=parameters['kind'],
                parameters=parameters['parameters']
            )

    optimizer: Description
    scheduler: Description
    grad_norm: Description


class Optimizer:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.step_idx = 0
        self._optimizer: torch.optim.Optimizer = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.state_dict_ = None

    def connect(self, parameters):
        parameters = list(parameters)

        self._optimizer = self.__make_optimizer__(parameters)
        self._scheduler = self.__make_scheduler__()

        self.__register_grad_norm_hooks__(parameters)

        if self.state_dict_ is not None:
            self.load_state_dict(self.state_dict_)

            del self.state_dict_

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self.step_idx += 1

        self._optimizer.step()

        if self._scheduler is not None:
            self._scheduler.step()

    @property
    def step_count(self):
        return self.step_idx

    @property
    def learning_rate(self):
        if self._scheduler is not None:
            return self._scheduler.get_last_lr()

        return self._optimizer.param_groups[0]['lr']

    @property
    def is_connected(self):
        return self._optimizer is not None

    def __make_optimizer__(self, parameters):
        cls = None

        match self.configuration.optimizer.kind:
            case 'adam':
                cls = torch.optim.Adam
            case 'adam_w':
                cls = torch.optim.AdamW
            case 'adamax':
                cls = torch.optim.Adamax
            case 'sgd':
                cls = torch.optim.SGD
            case 'asgd':
                cls = torch.optim.ASGD
            case 'rmsprop':
                cls = torch.optim.RMSprop
            case _:
                raise ValueError(f'Unknown optimizer kind: {self.configuration.optimizer.kind}')

        return cls(parameters, **self.configuration.optimizer.parameters)

    def __make_scheduler__(self):
        if self.configuration.scheduler is None:
            return None

        cls = None

        match self.configuration.scheduler.kind:
            case 'constant':
                cls = torch.optim.lr_scheduler.ConstantLR
            case 'multi_step':
                cls = torch.optim.lr_scheduler.MultiStepLR
            case 'linear':
                cls = torch.optim.lr_scheduler.LinearLR
            case 'exponential':
                cls = torch.optim.lr_scheduler.ExponentialLR
            case 'cosine':
                cls = torch.optim.lr_scheduler.CosineAnnealingLR
            case 'cosine_restarts':
                cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

        return cls(self._optimizer, **self.configuration.scheduler.parameters)

    def __register_grad_norm_hooks__(self, parameters):
        if self.configuration.grad_norm is None:
            return

        match self.configuration.grad_norm.kind:
            case 'norm':
                value = self.configuration.grad_norm.parameters['value']

                for p in parameters:
                    p.register_hook(lambda grad: torch.nn.functional.normalize(grad, dim=-1) * value)
            case 'clip':
                value = self.configuration.grad_norm.parameters['value']

                for p in parameters:
                    p.register_hook(lambda grad: torch.clamp(grad, -value, value))
            case _:
                raise ValueError('Unsupported value for grad norm')

    def state_dict(self):
        return dict(
            optimizer=self._optimizer.state_dict(),
            scheduler=self._scheduler.state_dict() if self._scheduler is not None else None,
            step_idx=self.step_idx
        )

    def load_state_dict(self, state_dict: dict):
        if self._optimizer is None:
            self.state_dict_ = state_dict
            return

        self._optimizer.load_state_dict(state_dict['optimizer'])

        if scheduler_state := state_dict['scheduler']:
            self._scheduler.load_state_dict(scheduler_state)

        self.step_idx = state_dict['step_idx']

    @staticmethod
    def from_cli(parameters):
        optimizer = parameters['model']
        optimizer = Configuration.Description.from_cli(optimizer)

        scheduler = parameters.get('scheduler')

        if scheduler is not None:
            scheduler = Configuration.Description.from_cli(scheduler)

        grad_norm = parameters.get('grad_norm')

        if grad_norm is not None:
            grad_norm = Configuration.Description.from_cli(grad_norm)

        return Optimizer(Configuration(optimizer=optimizer, scheduler=scheduler, grad_norm=grad_norm))
