"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)
Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""
from abc import ABC, abstractmethod
from prediction_properties import PredictionProperties


class BasePrompt(ABC):

    def __init__(self, system_identity=None, task=None, format_output=None):
        self.custom_system_identity = system_identity
        self.custom_task = task
        self.custom_format_output = format_output

    @abstractmethod
    def default_system_identity(self):
        pass

    @abstractmethod
    def default_task(self):
        pass

    @abstractmethod
    def default_format_output(self):
        pass

    def system_identity(self):
        if self.custom_system_identity is not None:
            return self.custom_system_identity
        return self.default_system_identity()

    def task(self):
        if self.custom_task is not None:
            return self.custom_task
        return self.default_task()

    def format_output(self):
        if self.custom_format_output is not None:
            return self.custom_format_output
        return self.default_format_output()

    def build(self):
        return self.system_identity(), self.task(), self.format_output()

class EntityExtractionPrompt(BasePrompt):

    def default_system_identity(self):
        return "You are a linguistic expert that specializes in identifying properties within a prediction statement."

    def default_task(self):
        return """For each word within the sentence "label" as either a "no_label": 0, "source": 1, "target": 2, "date": 3, "outcome": 4. IMPORTANT: Keep multi-word entities together as single items in the list."""

    def default_format_output(self):
        return """
        Respond ONLY with valid JSON in this exact format: {0: [word_from_sentence]}, {1: [word_from_sentence]}, {2: [word_from_sentence]}, {3: [word_from_sentence]}, {4: [word_from_sentence]}, where key is int ranging from 0 to 4 and the value is the words_from_sentence, split by a comma/all placed into a list, so {int: [word_from_sentence_1, word_from_sentence_2, ..., word_from_sentence_W]}. For 2 and 3, some words may be a prefix or a position or title before/after 2 or 3. Be sure to take that into account.
        Do NOT reason or provide anything other than the aforementioned. Also, stop responding in reverse format {word_from_sentence: 0}, {word_from_sentence: 1}, {word_from_sentence: 2}, {word_from_sentence: 3}, {word_from_sentence: 4} or in any other format.
        Respond ONLY with valid JSON in this exact format: {0: [word_from_sentence]}, {1: [word_from_sentence]}, {2: [word_from_sentence]}, {3: [word_from_sentence]}, {4: [word_from_sentence]}, where key is int ranging from 0 to 4 and the value is the words_from_sentence, split by a comma/all placed into a list, so {int: [word_from_sentence_1, word_from_sentence_2, ..., word_from_sentence_W]}.
        """

    def few_shot(self):
        source_ex = PredictionProperties.get_source_examples()
        target_ex = PredictionProperties.get_target_examples()
        date_ex = PredictionProperties.get_date_examples()
        outcome_ex = PredictionProperties.get_outcome_examples()

        few_shot_examples = f"""
        Here are examples of each property to guide you:
        - Source (1): {source_ex}
        - Target (2): {target_ex}
        - Date (3): {date_ex}
        - Outcome (4): {outcome_ex}
        """
        return self.system_identity(), self.task(), self.format_output(), few_shot_examples

class SentenceClassificationPrompt(BasePrompt):

    def default_system_identity(self):
        # TODO: Define system identity for sentence classification
        return ""

    def default_task(self):
        # TODO: Define classification task instructions
        return ""

    def default_format_output(self):
        # TODO: Define expected output format for sentence classification
        return ""

    def few_shot(self):
        # TODO: Add few shot examples for sentence classification
        few_shot_examples = ""
        return self.system_identity(), self.task(), self.format_output(), few_shot_examples