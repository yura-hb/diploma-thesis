import torch
import torch.nn as nn


# Taken from https://github.com/thomashirtz/noisy-networks/blob/main/noisynetworks.py


class AbstractNoisyLayer(nn.modules.lazy.LazyModuleMixin, nn.Module):

    def __init__(self, output_features: int, sigma: float):
        super().__init__()

        self.sigma = sigma
        self.input_features = None
        self.output_features = output_features

        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.mu_weight = nn.UninitializedParameter()
        self.sigma_weight = nn.UninitializedParameter()

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.input_features = input.shape[-1]

                self.mu_weight.materialize((self.output_features, self.input_features))
                self.sigma_weight.materialize((self.output_features, self.input_features))

                self.parameter_initialization()
                self.sample_noise()

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        return nn.functional.linear(x, weight=self.weight, bias=self.bias)

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bias(self) -> torch.Tensor:
        raise NotImplementedError

    def sample_noise(self) -> None:
        raise NotImplementedError

    def parameter_initialization(self) -> None:
        raise NotImplementedError

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.zeros(features).uniform_(-self.bound, self.bound).to(self.mu_bias.device)

        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class IndependentNoisyLayer(AbstractNoisyLayer):
    def __init__(self, output_features: int, sigma: float = 0.017):
        super().__init__(output_features=output_features, sigma=sigma)

        self.epsilon_bias = None
        self.epsilon_weight = None
        self.bound = None

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * self.epsilon_weight + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_bias + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_bias = self.get_noise_tensor((self.output_features,))
        self.epsilon_weight = self.get_noise_tensor((self.output_features, self.input_features))

    def parameter_initialization(self) -> None:
        self.bound = (3 / self.input_features) ** 0.5

        self.sigma_bias.data.fill_(self.sigma)
        self.sigma_weight.data.fill_(self.sigma)
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)


class FactorisedNoisyLayer(AbstractNoisyLayer):
    def __init__(self, output_features: int, sigma: float = 0.5):
        super().__init__(output_features=output_features, sigma=sigma)

        self.epsilon_input = None
        self.epsilon_output = None

        self.bound = None

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_output + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def parameter_initialization(self) -> None:
        self.bound = self.input_features ** (-0.5)

        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)