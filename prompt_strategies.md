# Prompting Strategies for Two Tasks

## Task 1: Projection vs Non-Projection

### Zero-Shot Prompt
```
You are a linguistic expert that specializes in identifying prediction statements, where a prediction statement is...

Requirements of a prediction:   
    1. Future orientation (Tense & Temporal Scope): The statement refers to a state or event occurring after the time of utterance.
       1.1. Linguistic Indicators: Modal verbs (will, shall, might), future-leaning verbs (forecast, project, estimate, expect, anticipate, prophecy, prognosis, guess, speculate, forecast, foretell), or temporal markers (tomorrow, next year, eventually). Implicit: modal verbs and future-leaning verbs. Explicit: “by 2030” or “next year”, etc. May use the future modal verb tense, such as: ['will', 'shall', 'would', 'going', 'might', 'should', 'could', 'may', 'must', 'can'].   
    2. Falsifiable Assertion: The statement must take a declarative claim that can be objectively verified as “True” or “False” once the timeframe is reached.
    3. Probabilistic Uncertainty: The statement reflects an estimate of likelihood rather than an established fact or a recorded history.
    4. Could also be a "past prediction", where the source is stating what they or someone predicted in the past.  
    
Sentence to label: '{sentence_to_classify}'
Classify the sentence as either a "prediction": 1 or "non-prediction": 0.
Respond ONLY with valid JSON in this exact format: {"predicted_sentence_label": 0} or {"predicted_sentence_label": 1}. Do NOT reason or provide anything other than {"predicted_sentence_label": 0} or {"predicted_sentence_label": 1}.
```

### Few-Shot Prompt

```
You are a linguistic expert that specializes in identifying prediction statements, where a prediction statement is...

Requirements of a prediction:   
    1. Future orientation (Tense & Temporal Scope): The statement refers to a state or event occurring after the time of utterance.
       1.1. Linguistic Indicators: Modal verbs (will, shall, might), future-leaning verbs (forecast, project, estimate, expect, anticipate, prophecy, prognosis, guess, speculate, forecast, foretell), or temporal markers (tomorrow, next year, eventually). Implicit: modal verbs and future-leaning verbs. Explicit: “by 2030” or “next year”, etc. May use the future modal verb tense, such as: ['will', 'shall', 'would', 'going', 'might', 'should', 'could', 'may', 'must', 'can'].   
    2. Falsifiable Assertion: The statement must take a declarative claim that can be objectively verified as “True” or “False” once the timeframe is reached.
    3. Probabilistic Uncertainty: The statement reflects an estimate of likelihood rather than an established fact or a recorded history.
    4. Could also be a "past prediction", where  the source is stating what they or someone predicted in the past.  
    
Examples:   
    1. {JPMorgan Chase forecasts that the net profit at Amazon potentially decrease in Q3 of 2027.: 1}   
    2. {Here are some of the shows, specials and movies coming to TV this week, Jan. 21-Feb. 6. : 1}
    
Sentence to label: '{sentence_to_classify}'

Classify the sentence as either a "prediction": 1 or "non-prediction": 0.

Respond ONLY with valid JSON in this exact format: {"predicted_sentence_label": 0} or {"predicted_sentence_label": 1}. Do NOT reason or provide anything other than {"predicted_sentence_label": 0} or {"predicted_sentence_label": 1}.

```

### Chain-of-Thought Prompt
```
You are a linguistic expert that specializes in identifying prediction statements, where a prediction statement is...
Requirements of a prediction:   
    1. Future orientation (Tense & Temporal Scope): The statement refers to a state or event occurring after the time of utterance.
       1.1. Linguistic Indicators: Modal verbs (will, shall, might), future-leaning verbs (forecast, project, estimate, expect, anticipate, prophecy, prognosis, guess, speculate, forecast, foretell), or temporal markers (tomorrow, next year, eventually). Implicit: modal verbs and future-leaning verbs. Explicit: “by 2030” or “next year”, etc. May use the future modal verb tense, such as: ['will', 'shall', 'would', 'going', 'might', 'should', 'could', 'may', 'must', 'can'].   
    2. Falsifiable Assertion: The statement must take a declarative claim that can be objectively verified as “True” or “False” once the timeframe is reached.
    3. Probabilistic Uncertainty: The statement reflects an estimate of likelihood rather than an established fact or a recorded history.
    4. Could also be a "past prediction", where the source is stating what they or someone predicted in the past.  

Sentence to label: '{sentence_to_classify}'

Follow these reasoning steps to classify the sentence:
   - Step 1: Analyze the sentence for future orientation and identify any linguistic indicators or temporal markers.
   - Step 2: Determine if the statement contains a falsifiable assertion.
   - Step 3: Evaluate if it reflects probabilistic uncertainty rather than established fact.
   - Step 4: Check if it represents a "past prediction."
   - Step 5: Synthesize your findings to classify the sentence as a "prediction": 1 or "non-prediction": 0.

Provide your reasoning for steps 1-5. Then, on a new line, output ONLY valid JSON in this exact format for your final classification:
{"predicted_sentence_label": 0, "reasoning": [insert your reasoning]} or {"predicted_sentence_label": 1, "reasoning": [insert your reasoning]}

```
---
```

### Evaluation Strategy

-   **Zero-shot:** Tests pure LLM knowledge and reasoning
-   **Few-shot:** Tests learning from minimal/3% examples
-   **Chain-of-thought:** Tests explicit reasoning capability

<!-- Compare performance across all 9 conditions (3 experiments × 3 prompting strategies) to determine optimal approach for pragmatic context modeling in low-resource MT. -->