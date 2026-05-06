from typing import Any

from owl.agent import Agent, KaggleObservation

AGENT: Agent | None = None


def agent_fn(observation: Any) -> list[list[float]]:
    global AGENT
    if AGENT is None:
        AGENT = Agent()

    return AGENT.act(KaggleObservation.model_validate(observation))
