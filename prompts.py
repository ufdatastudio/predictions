"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)
Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""
from abc import ABC, abstractmethod
from data_processing import DataProcessing
from prediction_properties import PredictionProperties


class BasePrompt(ABC):
    """
    Abstract base class for all prompt types.
    Implements Factory Method pattern for flexible prompt construction.
    Allows customization of system identity, task, and format output.
    """

    def __init__(self, system_identity=None, task=None, format_output=None, prompt_type_name=None):
        self.custom_system_identity = system_identity
        self.custom_task = task
        self.custom_format_output = format_output
        self.prompt_type_name = prompt_type_name

    @abstractmethod
    def default_system_identity(self):
        """Define the default system identity/role for the LLM."""
        pass

    @abstractmethod
    def default_task(self):
        """Define the default task the LLM should perform."""
        pass

    @abstractmethod
    def default_format_output(self):
        """Define the default output format expected from the LLM."""
        pass

    def get_prompt_name(self):
        """Returns the name/type of the prompt (zero-shot, few-shot, chain-of-thought)."""
        # print(self.prompt_type_name)
        return self.prompt_type_name

    def system_identity(self):
        """Returns custom system identity if provided, otherwise default."""
        if self.custom_system_identity is not None:
            return self.custom_system_identity
        return self.default_system_identity()

    def task(self):
        """Returns custom task if provided, otherwise default."""
        if self.custom_task is not None:
            return self.custom_task
        return self.default_task()

    def format_output(self):
        """Returns custom format output if provided, otherwise default."""
        if self.custom_format_output is not None:
            return self.custom_format_output
        return self.default_format_output()

    def default_steps(self):
        """
        Define default reasoning steps for chain-of-thought prompting.
        Currently returns system_identity as fallback; should be overridden in subclasses.
        """
        if self.custom_system_identity is not None:
            return self.custom_system_identity
        return self.default_system_identity()

    def build(self):
        """Basic build method returning the three core prompt components."""
        return self.system_identity(), self.task(), self.format_output()

    def zero_shot(self):
        """
        Zero-shot prompting: No examples provided.
        Returns system identity, task, and format output.
        """
        return self.system_identity(), self.task(), self.format_output()

    def few_shot(self):
        """
        Few-shot prompting: Provides examples for each TOLSA-M property.
        Returns system identity, task, format output, and examples.
        """
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

    def chain_of_thought(self):
        """
        Chain-of-thought prompting: Provides step-by-step reasoning instructions.
        Returns system identity, task, format output, and reasoning steps.
        """
        return self.system_identity(), self.task(), self.format_output(), self.default_steps()


class SentenceClassificationPrompt(BasePrompt):
    """
    Prompt for classifying sentences as TOLSA-M or non-TOLSA-M.
    Supports zero-shot, few-shot, and chain-of-thought approaches.
    """

    def default_system_identity(self):
        tolsa_m_definition = PredictionProperties.get_tolsa_m_definition()
        return f"""You are a linguistic expert that specializes in identifying TOLSA-M (Target Outcome with optionaL Source, dAte, and Metadata) from a given text input.
        {tolsa_m_definition}"""

    def default_task(self):
        return """Classify the sentence as either a "TOLSA-M": 1 or "non-TOLSA-M": 0."""

    def few_shot(self, dataset_path: str = None, stratify_columns: list = None, seed: int = 3):
        """
        Few-shot prompting: Provides examples for each TOLSA-M property.
        Returns system identity, task, format output, and examples.
        
        Parameters
        ----------
        dataset_path : str, optional
            Path to training data CSV file for few-shot examples
        stratify_columns : list of str, optional
            Columns to stratify by (e.g., ['Ground Truth', 'Dataset Name'])
            If 2 columns provided, uses balanced pair sampling for max diversity
        seed : int
            Random seed for reproducible sampling
        """
        if dataset_path:
            # Load training data
            train_df = DataProcessing.load_from_file(dataset_path, 'csv', sep=',', encoding='utf-8')
            
            # Default to stratifying by label only
            if stratify_columns is None:
                stratify_columns = ['Ground Truth']
            
            # Choose sampling strategy based on number of stratification columns
            if len(stratify_columns) == 2:
                # Balanced pair sampling: 1 pos + 1 neg from each dataset
                few_shot_df = DataProcessing.balanced_pair_sampling(
                    train_df,
                    label_column=stratify_columns[0],
                    dataset_column=stratify_columns[1],
                    n_samples=7,
                    random_state=seed
                )
            else:
                # Single-level stratification
                few_shot_df = DataProcessing.stratified_sample(
                    train_df,
                    label_column=stratify_columns[0],
                    n_samples=7,
                    random_state=seed
                )
            
            # Format examples for prompt
            few_shot_examples = "\n"
            for idx, row in few_shot_df.iterrows():
                sentence = row['Base Sentence']
                label = row['Ground Truth']
                label_name = "TOLSA-M" if label == 1.0 else "non-TOLSA-M"
                
                # Include dataset info for transparency
                dataset_info = ""
                if 'Dataset Name' in row:
                    dataset_info = f" [Source: {row['Dataset Name']}]"
                
                few_shot_examples += f"\n\t\tExample {idx+1}{dataset_info}: \"{sentence}\" → {label_name}\n\n"
            
            return self.system_identity(), self.task(), self.format_output(), few_shot_examples
        
        # Fallback to property examples if no dataset provided
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

    def default_format_output(self):
        # Matches the expected format in llm-classifiers.py parse_json_response
        if self.get_prompt_name() == 'zero-shot' or self.get_prompt_name() == 'few-shot':
            return """Respond ONLY with valid JSON: {"y_hat": 1} or {"y_hat": 0}. Do NOT include reasoning or additional text."""
        elif self.get_prompt_name() == 'chain-of-thought':
            return """Respond ONLY with valid JSON in this exact format: {"y_hat": 0, "reasoning": [insert your reasoning]} or {"y_hat": 1, "reasoning": [insert your reasoning]}. Be sure to reason and do NOT provide anything other than {"y_hat": 0, "reasoning": [insert your reasoning]} or {"y_hat": 1, "reasoning": [insert your reasoning]}."""

    def default_steps(self):
        """
        Chain-of-thought reasoning steps for TOLSA-M classification.
        Updated to handle past, present, and future tenses.
        """
        return """
        - Step 1: Identify the tense (past, present, or future) and check for temporal indicators or predictive language across all tenses.
        - Step 2: Determine if the statement contains a target entity and a measurable outcome (attribute, metric, or slope).
        - Step 3: Check for optional source entity and date information (declaration or fruition).
        - Step 4: Evaluate if the statement represents uncertainty, expectation, or a previously declared forecast (for past TOLSA-M).
        - Step 5: Verify the statement meets at least one indicator requirement (predictive language, temporal constructions, or attribution).
        - Step 6: Synthesize your findings to classify the sentence as a "TOLSA-M": 1 or "non-TOLSA-M": 0.
"""


class EntityExtractionPrompt(BasePrompt):
    """
    Prompt for extracting and labeling TOLSA-M entities from text.
    Identifies source, target, date, and outcome components.
    """

    def default_system_identity(self):
        tolsa_m_definition = PredictionProperties.get_tolsa_m_definition()
        return f"""You are a linguistic expert that specializes in identifying TOLSA-M (Target Outcome with optionaL Source, dAte, and Metadata) properties from a given text input.
        {tolsa_m_definition}
"""

    def default_task(self):
        return """For each word or span within the sentence, label it as either "source": 1, "target": 2, "date": 3, "outcome": 4. IMPORTANT: Keep multi-word spans together as single items in the list. Return [] for any property not present in the sentence. Only extract words that fit these 4 categories."""

    def few_shot(self):
        """
        Few-shot prompting for slot filling: Provides explicit sentence-to-JSON
        mapping examples using real TOLSA-M property examples from PredictionProperties.

        Returns
        -------
        tuple
            system_identity, task, format_output, few_shot_examples
        """
        source_ex  = PredictionProperties.get_source_examples()
        target_ex  = PredictionProperties.get_target_examples()
        date_ex    = PredictionProperties.get_date_examples()
        outcome_ex = PredictionProperties.get_outcome_examples()

        few_shot_examples = f"""
        Here are examples of how to map a sentence to the required JSON format.
        Key schema: {{"1": [source], "2": [target], "3": [date], "4": [outcome]}}

        Example 1 (finance — all properties present):
        Sentence: "Goldman Sachs predicts Apple stock will rise by 20% in Q3."
        Output: {{"1": ["{source_ex[0]}"], "2": ["{target_ex[0]}", "stock"], "3": ["{date_ex[10]}"], "4": ["{outcome_ex['slope'][0]}", "20%"]}}

        Example 2 (weather — source and date present):
        Sentence: "The National Weather Service expects temperatures at the Gulf Coast to rise sharply by 2025."
        Output: {{"1": ["{source_ex[5]}"], "2": ["{target_ex[7]}"], "3": ["{date_ex[8]}"], "4": ["{outcome_ex['slope'][4]}"]}}

        Example 3 (sports — no source):
        Sentence: "Simone Biles is expected to win in Q3."
        Output: {{"1": [], "2": ["{target_ex[4]}"], "3": ["{date_ex[10]}"], "4": ["{outcome_ex['attribute_of_interest'][1]}"]}}

        Example 4 (health — no date):
        Sentence: "Dr. Keith L. Black predicts the CDC heart rate monitoring program will decrease."
        Output: {{"1": ["{source_ex[9]}"], "2": ["{target_ex[12]}"], "3": [], "4": ["{outcome_ex['slope'][5]}"]}}

        Example 5 (non-TOLSA-M — return empty lists for all properties):
        Sentence: "The company held its annual meeting last Tuesday."
        Output: {{"1": [], "2": [], "3": [], "4": []}}
        Key reminders:
        - Source examples: {source_ex[:4]}
        - Target examples: {target_ex[:4]}
        - Date examples:   {date_ex[:4]}
        - Outcome examples (attribute): {outcome_ex['attribute_of_interest'][:3]}
        - Outcome examples (slope):     {outcome_ex['slope'][:3]}
        - Return [] for any property not present in the sentence.
        - Keep multi-word spans together as single list items.
        """

        return self.system_identity(), self.task(), self.format_output(), few_shot_examples

    def default_format_output(self):
        if self.get_prompt_name() == 'zero-shot' or self.get_prompt_name() == 'few-shot':
            return """Respond ONLY with valid JSON: {"1": [], "2": [], "3": [], "4": []}. Do NOT include reasoning or additional text. Return [] for any property not present in the sentence."""
        elif self.get_prompt_name() == 'chain-of-thought':
            return """Respond ONLY with valid JSON in this exact format: {"1": [], "2": [], "3": [], "4": [], "reasoning": "[insert your reasoning]"}. Be sure to reason and do NOT provide anything other than the 
            aforementioned format."""
