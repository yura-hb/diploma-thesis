
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
        super().__init__(*args, **kwargs)

        self.configuration = configuration
        self.trpo_loss = Loss(configuration=Loss.Configuration(kind='cross_entropy', parameters=dict()))

    def __train__(self, model: Policy):
        try:
            batch, generator = self.storage.sample_minibatches(update_returns=self.configuration.update_advantages,
                                                               device=self.run_configuration.device,
                                                               n=self.configuration.epochs,
                                                               sample_ratio=self.configuration.sample_ratio)

            for minibatch in generator:
                self.__step__(minibatch, model, self.configuration)

            # Load batch
            batch = batch()

            # Auxiliary step
            self.__auxiliary_step__(model, batch)
        except NotReadyException:
            return

    def __auxiliary_step__(self, model: Policy, batch: Batch):
        output = model(batch.state)

        assert Keys.ACTOR_VALUE in output.keys(), (f"Actor value not found in output. It should be a value "
                                                   f"representing value estimate for actor head")

        actor_values = output[Keys.ACTOR_VALUE]

        _, actions, _ = model.__fetch_values__(output)

        loss = self.configuration.value_loss(actor_values, batch.info[Record.RETURN_KEY])
        loss += self.configuration.trpo_penalty * self.trpo_loss(actions, batch.info[Record.POLICY_KEY])

        self.step(loss, self.optimizer)

        self.record_loss(loss, key='auxiliary')

    @classmethod
    def from_cli(cls,
                 parameters: Dict,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        schedule = TrainSchedule.from_cli(parameters)
        configuration = Configuration.from_cli(parameters)

        return cls(configuration=configuration,
                   memory=memory,
                   optimizer=optimizer,
                   loss=loss,
                   return_estimator=return_estimator,
                   train_schedule=schedule)
