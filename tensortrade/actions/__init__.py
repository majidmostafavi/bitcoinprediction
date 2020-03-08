from .action_scheme import ActionScheme, DTypeString, TradeActionUnion
from .continuous_actions import ContinuousActions
from .continuous_actions_plus import ContinuousActionsPlus
from .discrete_actions import DiscreteActions
from .discrete_actions_plus import DiscreteActionsPlus
from .multi_discrete_actions import MultiDiscreteActions
from .custome_discrete_actions import CustomeDiscreteActions


_registry = {
    'continuous': ContinuousActions,
    'continuous-plus': ContinuousActionsPlus,
    'discrete': DiscreteActions,
    'multi-discrete': MultiDiscreteActions,
    'custome-discrete': CustomeDiscreteActions,
    'discrete-plus': DiscreteActionsPlus,
}


def get(identifier: str) -> ActionScheme:
    """Gets the `ActionScheme` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `ActionScheme`

    Raises:
        KeyError: if identifier is not associated with any `ActionScheme`
    """
    if identifier not in _registry.keys():
        raise KeyError(f'Identifier {identifier} is not associated with any `ActionScheme`.')

    return _registry[identifier]()
