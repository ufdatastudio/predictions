from abc import ABC, abstractmethod

from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

from data_processing import DataProcessing
from text_generation_models import TextGenerationModelFactory

class TemplateRequirements(ABC):
    def __init__(self, domain, N):
        self.properties_template = None
        self.requirements_template = None

class PredictionTemplateRequirements(TemplateRequirements):
    def __init__(self, domain, N, domain_examples):

            self.properties_template = """A prediction <p> = (<p_s>, <p_t>, <p_d>, <p_o>), where it consists of the following four properties:
        
            1. <p_s>, any source entity in the {domain} domain.
                - Can be a person (with a name) or a {domain} person such as a {domain} reporter, {domain} analyst, {domain} expert, {domain} top executive, {domain} senior level person, etc), civilian.
                - Can only be an organization that is associated with the {domain} prediction.
            2. <p_t>, any target entity in the {domain} domain.
                - Can be a person (with a name) or a {domain} person such as a {domain} reporter, {domain} analyst, {domain} expert, {domain} top executive, {domain} senior level person, etc).
                - Can only be an organization that is associated with the {domain} prediction.
            3. <p_d>, date or time range when <p> is expected to come to fruition or when one should observe the <p>.
                - Forecast can range from a second to anytime in the future.
                - Answers the questions: "How far to go out from today?" or "Where to stop?".
            4. <p_o>, {domain} prediction outcome.
                - Details relevant details such as outcome, a quantifiable metric, or slope.
                - Some examples are {prediction_domain_outcome}."""
        
            self.requirements_template = """{domain} requirements to use for each prediction:
            - Should be based on real-world {domain} data and not hallucinate.
            - Only a simple sentence (prediction) (and NOT compounding using "and" or "or").
            - Should diversify all four properties of the prediction (<p>) as in change and not use same for <p_s>, <p_t>, <p_d>, <p_o>.
            - Should use synonyms to predict such as forecasts, speculates, foresee, envision, etc., and not use any of them more than ten times.
            - The prediction should be unique and not repeated.
            - Do not number the predictions.
            - Do not say, "As the {domain}, I will generate company-based {domain} predictions using the provided templates." or anything similar.
            - Use the five different templates and examples provided.
            - Change how the current date (<p_d>) written in the prediction with examples of (1) Wednesday, August 21, 2024; (2) Wed, August 21, 2024; (3) 08/21/2024; (4) 08/21/2024; (5) 21/08/2024; (6) 21 August 2024; (7) 2024/08/21; (8) 2024-08-21; (9) August 21, 2024; (10) Aug 21, 2024; (11) 21 August 2024, (12) 21 Aug 2024, Q3 of 2027, 2029 of Q3, etc (with removing day of week).
            {domain_requirements}
            - Do not say, "Here are {N} unique {domain} predictions based on the provided templates and examples:" in the prompt.
            - Do not use any of the examples in the prompt.
            - In front of every prodiction, put the template number in the format of "T1:", "T2:", etc. and do not number them like "1.", "2.", etc. Should have template number and generated prediction matching.
            - Do not put template number on line by itself. Always pair with a prediction.
            - Disregard brackets: "<>"
            - Should never say "Here are {N} unique {domain} predictions based on the provided templates and examples:" or "Note: I've made sure to follow the guidelines and templates provided, and generated unique predictions that meet the requirements."
            - Do not use person name of entity name more than once as in don't use name Joe as both the <p_s> and <p_t>, unless like Mr. Sach and Goldman Sach or Mr. Sam Walton and Sam's Club, etc.
            - The source entity (<p_s>) is rarely the same as the target entity (<p_t>) and if same, the <p_s> is making a prediction on itself in the <p_t>.
            - Should variate the slope of rise/increase/as much as, fall/decrease/as little as, change, stay stable, high/low chance/probability/degree of, etc.
            - Should variate the prediction verbs such as will, would, be going to, should, etc."""
            self.examples_template = """Here are some examples of {domain} predictions:{domain_examples} With the above (prediction with four properties, requirements, templates, and examples), generate a unique set of {N} predictions per template following the examples. Think from the perspective of an {domain} analyst, expert, top executive, or senior level person and even a college student, professional, research advisor, etc."""

    def langchain_structure(self):
        self.full_prediction_template = """{prediction_properties}

        {prediction_requirements}

        {prediction_templates}

        {prediction_examples}
        """
        return self.full_prediction_template
    def properties_to_langchain(self):
         return PromptTemplate.from_template(self.properties_template)
    def requirements_to_langchain(self):
         return PromptTemplate.from_template(self.requirements_template)  
    def template_to_langchain(self, examples_templates):
         return self.examples_templates
    def examples_to_langchain(self, examples):
         return self.examples
    def all_input_to_langchain_pipeline(self):
        prediction_input_prompts = [
        ("prediction_properties", self.properties_to_langchain),
        ("prediction_requirements", self.requirements_to_langchain),
        ("prediction_templates", self.template_to_langchain),
        ("prediction_examples", self.examples_to_langchain),
        ]
        

        pipeline_prompts = PipelinePromptTemplate(
            final_prompt=self.full_prediction_template, pipeline_prompts=prediction_input_prompts
        )
        return pipeline_prompts
    
class ObsertionTemplateRequirements(TemplateRequirements):
    def __init__(self, domain, N, domain_examples):

            self.properties_template = """An observation <o> = (<o_s>, <o_t>, <o_d>, <o_a>), where it consists of the following four properties:
            1. <o_s>, any source entity in the {domain} domain.
                - Can be a person (with a name) or a {domain} person such as a {domain} reporter, {domain} analyst, {domain} expert, {domain} top executive, {domain} senior level person, etc, civilian.
                - Can only be an organization that is associated with the {domain} observation.
            2. <o_t>, any target entity in the {domain} domain.
                - Can be a person (with a name) or a {domain} person such as a {domain} reporter, {domain} analyst, {domain} expert, {domain} top executive, {domain} senior level person, etc).
                - Can only be an organization that is associated with the {domain} observation.
            3. <o_d>, date or time range when <o> is expected to come to fruition or when one should observe the <o>.
                - Forecast can range from a second to anytime in the future.
                - Answers the questions: "How far to go out from today?" or "Where to stop?".
            4. <o_a>, {domain} observation output.
                - Characteristics of a domain-specific outputs such as various quantifiable metrics relevant to the {domain} domain.
        - Some examples are {observation_domain_output}."""
        
            self.requirements_template = """{domain} requirements to use for each observation:

                - Should be based on real-world {observation_domain} data and not hallucinate.
                - Must be a simple sentence (observation) (and NOT compounding using "and" or "or").
                - Should diversify all four properties of the observation (<o>) as in change and not use same for <o_s>, <o_t>, <o_d>, <o_a>.
                - The observation should be unique and not repeated.
                - Do not number the observations.
                - In front of every observation, put the template number in the format of "T1:", "T2:", etc. and do not number them like "1.", "2.", etc. Should have template number and generated prediction matching.
                - Must not generate, "template 1:..., template 2:..., etc" or anything similar and don't generate "T1:", "T2:", etc by itself.    
                - Must not generate, "Here are {N} unique observation based on the provided templates or anything similar.
                - Change how the current date (<o_d>) written in the observation with examples of (1) Wednesday, August 21, 2024; (2) Wed, August 21, 2024; (3) 08/21/2024; (4) 08/21/2024; (5) 21/08/2024; (6) 21 August 2024; (7) 2024/08/21; (8) 2024-08-21; (9) August 21, 2024; (10) Aug 21, 2024; (11) 21 August 2024, (12) 21 Aug 2024, Q3 of 2027, 2029 of Q3, etc.
                - Do not use any of the examples in the prompt.
                - Do not put template number on line by itself. Always pair with an observation.
                - Disregard brackets: "<>"
                - Do not use person name of entity name more than once as in don't use name Joe as both the <o_s> and <o_t>, unless like Mr. Sach and Goldman Sach or Mr. Sam Walton and Sam's Club, etc.
                - The source entity (<o_s>) is rarely the same as the target entity (<o_t>) and if same, the <o_s> is making a observation on itself in the <o_t>.
                - Should variate the slope of rose/increased/as much as, fell/decreased/as little as, changed, stayed stable, high/low chance/probability/degree of, etc.
                - Must be past tense as in already occurred and not future tense.
                - Must not use will, would, be going to, should, etc. in the observation.
                - Do not include "{domain} template 1:	"
                - Should variate the past tense prediction verbs such as observed, saw, noted, etc.."""
            self.examples_template = """Here are some examples of {domain} predictions:{domain_examples} With the above (prediction with four properties, requirements, templates, and examples), generate a unique set of {N} predictions per template following the examples. Think from the perspective of an {domain} analyst, expert, top executive, or senior level person and even a college student, professional, research advisor, etc."""

    def langchain_structure(self):
        self.full_observation_template = """{observation_properties}

        {observation_requirements}

        {observation_templates}

        {observation_examples}
        """
        return self.full_observation_template
    def properties_to_langchain(self):
         return PromptTemplate.from_template(self.properties_template)
    def requirements_to_langchain(self):
         return PromptTemplate.from_template(self.requirements_template)  
    def template_to_langchain(self, examples_templates):
         return self.examples_templates
    def examples_to_langchain(self, examples):
         return self.examples
    def all_input_to_langchain_pipeline(self):
        observation_input_prompts = [
        ("observation_properties", self.properties_to_langchain),
        ("observation_requirements", self.requirements_to_langchain),
        ("observation_templates", self.template_to_langchain),
        ("observation_examples", self.examples_to_langchain),
        ]
        

        pipeline_prompts = PipelinePromptTemplate(
            final_prompt=self.full_observation_template, pipeline_prompts=observation_input_prompts
        )
        return pipeline_prompts