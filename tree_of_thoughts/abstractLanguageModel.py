from abc import ABC, abstractmethod

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k, initial_prompt):
        pass

    @abstractmethod
    def evaluate_states(self, states, initial_prompt):
        pass

    @abstractmethod
    def generate_solution(self, initial_prompt, state):
        pass