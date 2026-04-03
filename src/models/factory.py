from typing import Any, Dict, Type

from src.models.base import BaseAgent


class ModelFactory:
    """
    Registry for Multi-Agent Reinforcement Learning models.

    Allows for decoupled instantiation of different architectural
    approaches while maintaining a single unified Trainer interface.
    """

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]):
        cls._registry[name] = agent_class
        print(f'[ModelFactory] Registered algorithm: {name}')

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseAgent:
        if name not in cls._registry:
            raise ValueError(
                f"Algorithm '{name}' not found in registry. "
                f'Available: {list(cls._registry.keys())}'
            )

        agent_class = cls._registry[name]
        return agent_class(config)
