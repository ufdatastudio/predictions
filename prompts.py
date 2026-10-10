"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)
Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""
from abc import ABC, abstractmethod
from data_processing import DataProcessing
from tolsa_properties import Tolsa


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
        Few-shot prompting: Provides examples for each TOLSA property.
        Returns system identity, task, format output, and examples.
        """
        source_ex = Tolsa.get_source_examples()
        target_ex = Tolsa.get_target_examples()
        date_ex = Tolsa.get_date_examples()
        outcome_ex = Tolsa.get_outcome_examples()

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
    Prompt for classifying text documents as TOLSA or non-TOLSA.
    Supports zero-shot, few-shot, and chain-of-thought approaches.
    Built on the TOLSA definition from Tolsa.get_tolsa_definition().
    """

    def default_system_identity(self):
        tolsa_definition = Tolsa.get_tolsa_definition()
        return f"""You are a linguistic expert that specializes in classifying whether a given text document is a TOLSA or a non-TOLSA.
        {tolsa_definition}"""

    def linguistic_cues(self):
        """
        Linguistic cues for classification.

        Tolsa.get_linguistic_cues() opens with a note written for extraction
        ("They are not extraction requirements. Extract ..."). For classification we keep the
        tense and keyword lists from Tolsa but replace that opening note with an
        identification note.
        """
        cues = Tolsa.get_linguistic_cues()

        # The opening note ends at the first blank line; everything after it is the cue lists.
        header, separator, body = cues.partition("\n\n")

        if not separator:
            raise ValueError(
                "Unexpected format from Tolsa.get_linguistic_cues(): "
                "could not find the opening note to replace."
            )

        body = body.replace(
            "necessary but NOT sufficient",
            "potential cue, but neither necessary nor sufficient"
        )


        if not separator:
            raise ValueError(
                "Unexpected format from Tolsa.get_linguistic_cues(): "
                "could not find the opening note to replace."
            )

        classification_header = """
        Linguistic Cues for TOLSA Identification:

        NOTE: No single cue below is sufficient on its own, and none is required.
        TOLSA identification prioritizes semantic criteria and properties (target + measurable outcome) over keyword matching.
        Tense alone is neither necessary nor sufficient for TOLSA identification. TOLSAs may be expressed in past, present, or future tense. Future-related verb constructions are potential cues only, not requirements.
        Predictive keywords (KP), modal verbs (KMV), and time expressions (KTE) are also potential cues, not requirements. Time expressions provide date context rather than independently determining the classification.
        """

        return f"{classification_header}\n\n{body}"

    def properties_and_cues(self):
        """
        Returns (TOLSA properties, linguistic cues) for building the classification base prompt.
        """
        return Tolsa.get_tolsa_properties(), self.linguistic_cues()

    def default_task(self):
        return """Classify the text document as either a "TOLSA": 1 or "non-TOLSA": 0.

            Classify as "TOLSA": 1 only if the text document explicitly contains BOTH:
                1. a target (entity, event, or location of interest), and
                2. a measurable outcome (attribute, metric, or slope) about that target whose
                   correctness may be unknown at the time of declaration.
            The source and the date are optional. They give context but do not decide the label.

            Decide from the meaning of the text document, not from single cues:
                - Tense does not decide the label. A past, present, or future tense text document
                  can be a TOLSA, and future tense alone does not make one.
                - Predictive keywords, modal verbs, and time expressions are not required, and
                  none of them alone makes a TOLSA.
                - A text document whose outcome was already known or decided at the time of
                  declaration (for example, a completed or decided fact) is "non-TOLSA": 0.
                  A past tense text document whose outcome was still unknown at the time of
                  declaration (for example, a past prediction) can be a TOLSA.

            If the target or the measurable outcome is not explicitly stated, classify as
            "non-TOLSA": 0.
            """

    def few_shot(self, dataset_path: str = None, stratify_columns: list = None, seed: int = 3):
        """
        Few-shot prompting: Provides examples for each TOLSA property.
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
                label_name = "TOLSA" if label == 1.0 else "non-TOLSA"
                
                # Include dataset info for transparency
                dataset_info = ""
                if 'Dataset Name' in row:
                    dataset_info = f" [Source: {row['Dataset Name']}]"
                
                few_shot_examples += f"\n\t\tExample {idx+1}{dataset_info}: \"{sentence}\" → {label_name}\n\n"
            
            return self.system_identity(), self.task(), self.format_output(), few_shot_examples
        
        # Fallback to property examples if no dataset provided
        source_ex = Tolsa.get_source_examples()
        target_ex = Tolsa.get_target_examples()
        date_ex = Tolsa.get_date_examples()
        outcome_ex = Tolsa.get_outcome_examples()

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
        Chain-of-thought reasoning steps for TOLSA classification.
        Follows the TOLSA definition: required target and measurable outcome, any tense,
        optional source and date, and no single cue decides the label.
        """
        return """
        - Step 1: Identify the target: the entity, event, or location of interest that the text document is about. If no target is explicitly stated, it is "non-TOLSA": 0.
        - Step 2: Identify the measurable outcome stated about the target (attribute, metric, or slope). If no measurable outcome is explicitly stated, it is "non-TOLSA": 0.
        - Step 3: Decide whether the outcome's correctness may be unknown at the time of declaration. Tense does not matter: a past, present, or future tense text document can qualify, such as a past prediction. If the outcome was already known or decided at the time of declaration, it is "non-TOLSA": 0.
        - Step 4: Note the optional source (declaring entity) and date (declaration or fruition timing). They give context but do not decide the label.
        - Step 5: Do not decide from a single cue. Tense, predictive keywords, modal verbs, and time expressions are neither required nor sufficient on their own.
        - Step 6: Synthesize your findings to classify the text document as a "TOLSA": 1 or "non-TOLSA": 0.
"""


class EntityExtractionPrompt(BasePrompt):
    """
    Prompt for extracting and labeling TOLSA entities from text.
    Identifies source, target, date, and outcome components.
    """

    def default_system_identity(self):
        tolsa_definition = Tolsa.get_tolsa_definition()
        return f"""You are a linguistic expert that specializes in extracting TOLSA (Target Outcome with optionaL Source, dAte) properties from a given text input.
        {tolsa_definition}
"""

    def default_task(self):
        return """Extract the source, target, date, and outcome properties from the text document and
            place each extracted span in the correct JSON list:
            - "1" = source
            - "2" = target
            - "3" = date
            - "4" = outcome

            The goal is to extract the properties that are explicitly present in the text document,
            not to first decide whether the text document is a TOLSA. This is an extraction task,
            not a classification task. Extract each source, target, date, and outcome span that is
            explicitly supported by the text document.
                        
            Extraction requirements:
                1. Put each distinct extracted span in its own string in the list.
                2. Keep multi-word spans together as one string.
                3. Do NOT join multiple spans with "|". The output must contain separate list items; "|"
                is added later by the processing pipeline.
                4. Preserve the original order of extracted spans according to their occurrence in the
                input text document. Do not reorder spans within any property list. For example, if the text document contains "word1"
                    before "word2" before "word3", the output must be ["word1", "word2", "word3"],
                    not ["word2", "word1", "word3"].
                5. Copy each extracted span exactly as it appears in the input text document.
                6. Preserve the exact wording, spelling, capitalization, whitespace within the span,
                punctuation, and symbols appearing in the extracted span.
                7. Preserve symbols such as %, °, $, +, -, /, parentheses, commas, decimal points,
                and other symbols exactly as they appear when they are part of the extracted span.
                8. Do not normalize, simplify, remove, or replace symbols or formatting.
                9. Do not invent, infer, or paraphrase a span. Every extracted span must be a
                verbatim span from the input text document.
                10. Return [] for any property that is not explicitly stated in the text document.
                    Do not leave a property empty merely because the text document may not be a TOLSA.
                11. Extract target and outcome spans whenever they are explicitly stated, regardless
                    of whether the text document would qualify as a TOLSA.
                12. Only extract spans that belong to source, target, date, or outcome.
            """

    def few_shot(self):
        """
        Few-shot prompting for property extraction: Provides seven explicit
        sentence-to-JSON mapping examples that demonstrate the expected
        extraction behavior.

        Returns
        -------
        tuple
            system_identity, task, format_output, few_shot_examples
        """
        few_shot_examples = """
        Here are examples of how to map a sentence to the required JSON format.
        Key schema: {"1": [source], "2": [target], "3": [date], "4": [outcome]}

        Text Document 1 (non-TOLSA — FP: Tense and Modal Verb):
        Sentence: "Neither major U.S. political party will hold conventions or indeed primaries to select their 2012 Presidential nominees."
        Output: {"1": [], "2": ["major U.S. political party", "conventions", "primaries", "Presidential nominees"], "3": ["2012"], "4": []}

        Text Document 2 (non-TOLSA — FP: Predictive Keyword):
        Sentence: "I have a class project."
        Output: {"1": ["I"], "2": [], "3": [], "4": []}

        Text Document 3 (non-TOLSA — FP: Modal Verb):
        Sentence: "After this purchase, Cramo will become the second largest rental services provider in the Latvian market."
        Output: {"1": [], "2": ["Cramo", "Latvian market"], "3": [], "4": []}

        Text Document 4 (non-TOLSA — FP: Time Expression):
        Sentence: "Start by picking one word that captures your values and intentions for the next year."
        Output: {"1": [], "2": ["your values and intentions"], "3": ["the next year"], "4": []}

        Text Document 5 (TOLSA — FN: Tense and Time Expression):
        Sentence: "Charles Barkley predicted the Knicks would win the Eastern Conference Finals..."
        Output: {"1": ["Charles Barkley"], "2": ["Knicks"], "3": [], "4": ["win the Eastern Conference Finals"]}

        Text Document 6 (TOLSA — FN: Predictive Keyword, Modal Verb, Time Expression):
        Sentence: "Sports Rage's Gabe Morency says the Seattle Seahawks win"
        Output: {"1": ["Sports Rage's Gabe Morency"], "2": ["Seattle Seahawks"], "3": [], "4": ["win"]}

        Text Document 7 (TOLSA — FN: Predictive Keyword and Modal Verb):
        Sentence: "So, who makes it to the Final Four this time around? Who cuts down the nets?"
        Output: {"1": [], "2": ["who", "Final Four", "Who"], "3": ["this time around"], "4": ["makes it", "cuts down the nets"]}

    Key reminders:
        - Copy every extracted span verbatim from the text document,
        including punctuation and capitalization.
        - Outcome is a single property. Attribute, slope, and metric spans are all part of
        the Outcome property when they are explicitly present.
        - If multiple outcome spans are present, put each span in its own string in the
        Outcome list and preserve their order of occurrence in the text document.
        - Non-TOLSA documents may still contain extractable property spans — the
        absence of a measurable outcome is what disqualifies them as TOLSA.
        - Do not invent, infer, or paraphrase.
        - Return [] when a property is not explicitly stated.
        - If target or outcome is not explicitly stated, return [] for that property.
        """

        return self.system_identity(), self.task(), self.format_output(), few_shot_examples

    def default_format_output(self):
        if self.get_prompt_name() == 'zero-shot' or self.get_prompt_name() == 'few-shot':
            return """Respond ONLY with valid JSON in this exact format:
                {"1": [], "2": [], "3": [], "4": []}

                Each property value must be a list of strings.
                Put each extracted span in its own list item.
                Do NOT use "|" to join spans.
                Do NOT include reasoning or additional text.
                Return [] for any property not explicitly stated in the text document.
            """
        elif self.get_prompt_name() == 'chain-of-thought':
            return """Respond ONLY with valid JSON in this exact format and key order:
        {"reasoning": "", "1": [], "2": [], "3": [], "4": []}

        Fill in "reasoning" FIRST, then the four lists, so that the lists follow from your reasoning.

        Rules for "reasoning":
        - Keep it to 1-3 short sentences on a single line.
        - Do NOT use double quotes, line breaks, or backslashes inside it. Use single quotes if you need to refer to text.
        - Do NOT put the final spans or "|" in it as a substitute for the required values in keys "1", "2", "3", and "4".

        Rules for the lists:
        - Each property value for "1", "2", "3", and "4" must be a list of strings.
        - "1" = source
        - "2" = target
        - "3" = date
        - "4" = outcome
        - Put each extracted span in its own list item.
        - Keep multi-word spans together as one string.
        - Do NOT use "|" to join spans.
        - Preserve the original order of extracted spans within each property list.
        - Copy each extracted span exactly as it appears in the text document.
        - Preserve capitalization, punctuation, whitespace within the span, and symbols.
        - Do NOT normalize, simplify, remove, or replace symbols or formatting.
        - Do NOT invent, infer, or paraphrase spans.
        - Return [] for any property not explicitly stated in the text document.
        - If target or outcome is not explicitly stated, return [] for that property.

        Output only the JSON object: no text before or after it and no code fences.
        """

    def default_steps(self):
        """
        Chain-of-thought reasoning steps for TOLSA property extraction.
        Overrides BasePrompt.default_steps, which would otherwise repeat the system identity.
        """
        return """
        - Step 1: Read the text document and identify the target: the entity, event, or location of interest that the text document is about. If no target is explicitly stated, "2" stays [].
        - Step 2: Identify the outcome: each attribute, metric, or slope explicitly stated about the target. If no outcome is explicitly stated, "4" stays [].
        - Step 3: Identify the source (the declaring entity) and the date (declaration or fruition timing) only if they are explicitly stated. Otherwise "1" and "3" stay [].
        - Step 4: Check every span. It must be copied verbatim from the text document (exact wording, capitalization, punctuation, and symbols), with nothing invented, inferred, or paraphrased.
        - Step 5: Put each span in its own string in the correct list ("1" = source, "2" = target, "3" = date, "4" = outcome), in order of occurrence. Do not join spans with "|".
        - Step 6: Summarize steps 1-5 in a short reasoning (1-3 sentences, one line, no double quotes), then give the four lists.
"""