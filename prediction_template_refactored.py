from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

full_prediction_template = """{prediction_properties}

{prediction_requirements}

{prediction_templates}

{prediction_examples}"""
full_prediction_prompt = PromptTemplate.from_template(full_prediction_template)

prediction_properties_template = """$\hat{y}$, prediction
    - $\hat{y}_{p}$, person that predicted $\hat{y}$
        - Can be a person, a noun (such as a reporter, analyst, expert, top executive, senior level person, etc.)
    - $\hat{y}_{o}$, organization
        - Can only be an organization or entity that is associated with the prediction
    - $\hat{y}_{t}$, time when $\hat{y}$ was made
        - Time is the exact moment that can be measured in day, hour, minute, second, etc.
    - $\hat{y}_{f}$, forecast time when $\hat{y}$ is expected to come to fruition
        - Forecast can be from a second to anytime in the future.
        - Answers the questions: "How far to go out?" or "Where to stop?"
    - $\hat{y}_{a}$, prediction attribute
        - Measurable domain-specific attributes such as various quantifiable metrics relevant to the specific field of interest.
    - $\hat{y}_{m}$, prediction metric outcome
        - How much will the $\hat{y}_{a}$ rise/increase, fall/decrease, change, stay stable, high/low chance/probability/degree of, etc?
    - $\hat{y}_{v}$, future verb tense
        - A verb that is associated with the future such as will, would, be going to, should, etc."""
prediction_properties_prompt = PromptTemplate.from_template(prediction_properties_template)

prediction_requirements_template = """Requirements to use for each prediction
    1. Should be based on real-world data and not hallucinate.
    2. Only a simple sentence (prediction) (and NOT compounding using "and" or "or").
    3. Should be either positive, negative, or neutral for metric outcome ($\hat{y}_{m}$).
    4. Should diversity the metric outcome ($\hat{y}_{m}$).
    5. Diversity the name of person ($\hat{y}_{p}$) and name of organization ($\hat{y}_{o}$).
    6. Should use synonyms of predicts such as forecasts, speculates, forsee, envision, etc and not use any of them more than ten times.
    7. The prediction should be unique and not repeated
    8. The $\hat{y}_{f}$ should always be after $\hat{y}_{t}$
    9. Do not number the preditions
    10. Do not say, "As the Chief Financial Officer at a publicly traded company on the US Stock Exchange, I will generate five company-based financial predictions using the provided templates." or "As the Chief Meteorologist at a national weather forecasting agency, I will generate five weather-based predictions using the provided templates.", "As the Chief Financial Officer at a publicly traded company on the US Stock Exchange, I will generate five company-based financial predictions using the provided templates.", or "Company-Based Financial Predictions", anything similar, or anything that is not a prediction and only include the predictions without "Here are 10 company-based financial prediction..." or anything similar and without the numbers in front."""
prediction_requirements_prompt = PromptTemplate.from_template(prediction_requirements_template)

prediction_templates_template = """- Template 1: On [ $\hat{y}_{t}$,], [$\hat{y}_{p}$ ] predicts that the [ $\hat{y}_{a}$] at [ $\hat{y}_{o}$ ] [ $\hat{y}_{v}$] [ $\hat{y}_{m}$] by [$\hat{y}_{m}$, ] in [ $\hat{y}_{f}$].
- Template 2: On [ $\hat{y}_{t}$ ], [ $\hat{y}_{p}$ ] from [ $\hat{y}_{o}$ company name ] predicts that the [ $\hat{y}_{a}$ ] [ $\hat{y}_{v}$ ] by [ $\hat{y}_{m}$ ] in [ $\hat{y}_{f}$ ].
- Template 3: [ $\hat{y}_{p} $] predicts on [ $\hat{y}_{t}$ ] that the [ $\hat{y}_{a}$ ] at [ $\hat{y}_{o}$ ] [ $\hat{y}_{v}$ ] by [ $\hat{y}_{m}$ ] in [ $\hat{y}_{f}$ ].
- Template 4: According to [ $\hat{y}_{p}$ ] from [ $\hat{y}_{o}$ ], on [ $\hat{y}_{t}$ ], the [ $\hat{y}_{a}$ ] [ $\hat{y}_{v}$ ] by [ $\hat{y}_{m}$ ] in the timeframe of [ $\hat{y}_{f}$ ].
- Template 5: In [ $\hat{y}_{f}$ ], the [ $\hat{y}_a$ ] at [ $\hat{y}_{o}$ ] is expected to [ $\hat{y}_{v}$ ] by [ $\hat{y}_{m}$ ], as predicted by [ $\hat{y}_{p}$ ] on [ $\hat{y}_{t}$ ]."""
prediction_templates_prompt = PromptTemplate.from_template(prediction_templates_template)

# print(
#     pipeline_prompt.format(
#         person="Elon Musk",
#         example_q="What's your favorite car?",
#         example_a="Tesla",
#         input="What's your favorite social media site?",
#     )
# )

# print(
#     pipeline_prompt.format(
#         prediction_properties="Elon Musk",
#         prediction_requirements="What's your favorite car?",
#         prediction_templates="Tesla",
#         input="Generate 5 prediction sentences.",
#     )
# )