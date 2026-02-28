class Prompts: 

    def extract_prediction_properties():
        system_identity_prompt = "You are a lingustic expert that specializes in identifying properties within a prediction statement."
    # prediction_requirements = PredictionProperties.get_requirements()
        task = """For each word within the sentence "label" as either a "no_label": 0, "source": 1, "target": 2, "date": 3, "outcome": 4. IMPORTANT: Keep multi-word entities together as single items in the list."""
        sentence_label_format_output = """

        Respond ONLY with valid JSON in this exact format: {0: [word_from_sentence]}, {1: [word_from_sentence]}, {2: [word_from_sentence]}, {3: [word_from_sentence]}, {4: [word_from_sentence]}, where key is int ranging from 0 to 4 and the value is the words_from_sentence, split by a comma/all placed into a list, so {int: [word_from_sentence_1, word_from_sentence_2, ..., word_from_sentence_W]}. For 2 and 3, some words may be a prefix or a position or tile before/after 2 or 3. Be sure to take that into account.

        Do NOT reason or provide anything other than the aforementioned. Also, stop responding in reverse format {word_from_sentence: 0}, {word_from_sentence: 1}, {word_from_sentence: 2}, {word_from_sentence: 3}, {word_from_sentence: 4} or in any other format.

        Respond ONLY with valid JSON in this exact format: {0: [word_from_sentence]}, {1: [word_from_sentence]}, {2: [word_from_sentence]}, {3: [word_from_sentence]}, {4: [word_from_sentence]}, where key is int ranging from 0 to 4 and the value is the words_from_sentence, split by a comma/all placed into a list, so {int: [word_from_sentence_1, word_from_sentence_2, ..., word_from_sentence_W]}.
        """

        return system_identity_prompt, task, sentence_label_format_output