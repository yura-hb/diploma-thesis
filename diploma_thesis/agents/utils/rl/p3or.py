
from agents.utils.memory import NotReadyException
from agents.utils.policy.policy import Keys

from .utils.ppo_mixin import *


@dataclass
class Configuration(PPOConfiguration):
    trpo_penalty: bool

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            trpo_penalty=parameters.get('trpo_penalty', 0.1),
            **PPOConfiguration.base_parameters_from_cli(parameters),
        )


class P3OR(PPOMixin):

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(configuration, *args, **kwargs)

        self.configuration: Configuration = configuration
        self.trpo_loss = Loss(configuration=Loss.Configuration(kind='cross_entropy', parameters=dict()))

    def __train__(self, model: Policy):
        try:
            batch, generator, info = self.storage.sample_minibatches(device=self.device,
                                                                     n=self.configuration.epochs,
                                                                     sample_count=self.configuration.sample_count)

            for minibatch in generator:
                self.__step__(minibatch, model)

                # Auxiliary step
                self.__auxiliary_step__(model, minibatch)

            self.__increase_memory_priority__(info)
        except NotReadyException:
            return

    def __auxiliary_step__(self, model: Policy, batch: Batch):
        def compute_loss():
            output = model(batch.state)

            assert Keys.ACTOR_VALUE in output.keys(), (f"Actor value not found in output. It should be a value "
                                                       f"representing value estimate for actor head")

            actor_values = output[Keys.ACTOR_VALUE]

            _, actions, _ = model.__fetch_values__(output)

            pad_columns = batch.info[Record.POLICY_KEY].shape[-1] - actions.shape[-1]

            if pad_columns > 0:
                actions = torch.nn.functional.pad(actions, (0, pad_columns), "constant", torch.finfo(torch.float32).min)

            loss = self.configuration.value_loss(actor_values, batch.info[Record.RETURN_KEY])
            loss += self.configuration.trpo_penalty * self.trpo_loss(actions, batch.info[Record.POLICY_KEY])

            return loss, ()

        loss, _ = self.step(compute_loss, self.optimizer)

        self.record_loss(loss, key='auxiliary')

    @classmethod
    def from_cli(cls, parameters: Dict, **kwargs):
        schedule = TrainSchedule.from_cli(parameters)
        configuration = Configuration.from_cli(parameters)

        return cls(configuration=configuration, train_schedule=schedule, **kwargs)
