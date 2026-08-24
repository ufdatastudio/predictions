"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

This module defines the properties, requirements, indicators, and examples
for TOLSA-M (Target Outcome with optionaL Source, dAte, and Metadata) identification.
"""


class PredictionProperties:
    """
    Centralized class containing all TOLSA-M properties, requirements, indicators,
    and examples across different domains and tenses.
    """

    def get_tolsa_m_definition():
        """
        Returns the core definition of TOLSA-M.
        Used in system identity prompts and documentation.
        """
        tolsa_m_definition = """TOLSA-M is defined as a proposal of any tense (past, present, future) document with properties—must contain some target (the entities of interests within the document) and measurable outcomes (either being an attribute, metric, or slope) with optional source (the entity declaring the document) and date (of declaration and of fruition)—while possibly exhibiting some uncertainty about the outcome."""
        return tolsa_m_definition

    def get_required_properties():
        """
        Returns the two REQUIRED properties for TOLSA-M identification.
        A statement MUST contain both target and outcome to qualify as TOLSA-M.
        """
        required_properties = """Required Properties (MUST be present):

            1. <target>
                - Defined as:
                    - Target entity that the TOLSA-M is about.
                - Characteristics:
                    - A person with either: a name only, profile name only, gender only,
                      domain-specific title only, or any combination of these.
                    - May include an associated organization.

            2. <outcome>
                - Defined as:
                    - Outcome of the TOLSA-M.
                - Characteristics:
                    - Comprised of one or more of the following:
                        - Attribute of interest
                        - Quantifiable metric
                        - Trend or slope
                    - May also include:
                        - Value at a specific instant
                        - Statistical extrema (e.g., minimum or maximum)
                        - Change over an interval
                        - Second-order effect comparison
                        - Recurrent or cyclical pattern
"""
        return required_properties

    def get_optional_properties():
        """
        Returns the three OPTIONAL properties for TOLSA-M identification.
        These provide additional context but are not required for classification.
        """
        optional_properties = """Optional Properties (provide context):

            1. <source>
                - Defined as:
                    - Source entity that conveys the TOLSA-M.
                - Characteristics:
                    - A person with either: a name only, profile name only, gender only,
                      domain-specific title only, or any combination of these.
                    - May include an associated organization.

            2. <date>
                - Defined as:
                    - Date when the TOLSA-M is declared.
                    - Date when the TOLSA-M is expected to come into fruition.
                - Characteristics:
                    - Can have two distinct values based on the definition above.
                    - May answer the question: "How far out is the TOLSA-M from today?"
                    - Any standard or non-standard date format is acceptable.

            3. <metadata>
                - Defined as:
                    - Additional contextual information about the TOLSA-M.
                - Characteristics (Taxonomy Dimensions):
                    - Domain: The primary area of study for the TOLSA-M (finance, sports, weather, policy, health, other)
                    - Epistemic: The range of certainty or confidence of the TOLSA-M (speculative, probabilistic, assertive, certain)
                    - Tense Validity: Temporal framing of when the TOLSA-M was declared relative to present (past, present, future)
                    - Dialogue Act Type: The communication function of the TOLSA-M (question, assertion, commitment)
                    - Outcome Type: A scale/range for the measurable attribute of the TOLSA-M (binary, categorical, ordinal, value point, value interval, trend)
                    - Document/Representation: The textual unit containing the TOLSA-M (phrase/clause, sentence, span, paragraph)
                    - Context: Whether the TOLSA-M requires surrounding text to be fully understood (without context: self-contained in one sentence, with context: requires multiple sentences or prior/surrounding text)
"""
        return optional_properties

    def get_tolsa_m_properties():
        """
        Returns the complete formal definition of all TOLSA-M properties.
        Combines required (target, outcome) and optional (source, date, metadata) properties.
        Ordering reflects requirement hierarchy: target and outcome first, then optional context.
        """
        required = PredictionProperties.get_required_properties()
        optional = PredictionProperties.get_optional_properties()
        
        tolsa_m_properties = f"""A TOLSA-M = (<target>, <outcome>, <source>, <date>, <metadata>)

{required}

{optional}
"""
        return tolsa_m_properties

    def get_requirements():
        """
        Returns comprehensive indicators and requirements for identifying TOLSA-M.
        Updated to align with definition covering past, present, and future tenses.
        At least ONE indicator should be present for a statement to qualify as TOLSA-M.
        
        Key Distinction:
        - **Indicators** = Explicit predictive/forecasting words (e.g., "forecast", "projection", 
          "expects", "was expected", "had forecasted") - content words signaling prediction
        - **Tense verbs** = Grammatical constructions showing time/modality (e.g., "will", "might", 
          "should", "was expected to", "is projected to") - structural helpers indicating temporal orientation
        
        Example: "Goldman Sachs forecasts Apple will rise"
                 - "forecasts" = indicator
                 - "will" = tense verb
        """
        # Past-oriented verb constructions for past TOLSA-M
        past_tense_verbs = [
            "was expected to", "had forecasted", "would have",
            "was going to", "was to", "was projected to",
            "had anticipated", "was predicted to"
        ]

        # Present-oriented verb constructions
        present_tense_verbs = [
            "is expected to", "is forecast to", "is projected to",
            "is anticipated to", "is predicted to"
        ]

        # Future-oriented verb constructions
        future_tense_verbs = [
            "will", "shall", "would", "going to", "might",
            "should", "could", "may", "must", "can"
        ]

        # Predictive/estimation language for past tense (previously declared)
        past_tolsa_m_indicators = [
            "was expected", "had forecasted", "previously predicted",
            "was projected", "anticipated that", "had estimated",
            "forecasted", "predicted earlier", "prior forecast",
            "earlier projection", "had anticipated"
        ]

        # Predictive/estimation language for present tense
        present_tolsa_m_indicators = [
            "expects", "forecasts", "predicts", "projects",
            "estimates", "anticipates", "is forecasting",
            "is projecting", "is predicting"
        ]

        # Predictive/estimation language for future tense
        # Remove duplicates that belong in present tense
        future_tolsa_m_indicators = [
            "forecast", "projection", "estimate", "outlook",
            "expectation", "anticipation", "prognosis", "guess",
            "speculation", "forecasting", "foretelling",
            "forecasted outcome", "forecast estimate", "will predict",
            "speculates"
        ]
        
        # Remove duplicates across all indicator lists programmatically
        past_set = set(past_tolsa_m_indicators)
        present_set = set(present_tolsa_m_indicators)
        future_set = set(future_tolsa_m_indicators)
        
        # Ensure no overlap: prioritize past > present > future
        present_set = present_set - past_set
        future_set = future_set - past_set - present_set
        
        # Convert back to sorted lists for consistent output
        past_tolsa_m_indicators = sorted(list(past_set))
        present_tolsa_m_indicators = sorted(list(present_set))
        future_tolsa_m_indicators = sorted(list(future_set))

        tolsa_m_requirements = f"""Indicators of a TOLSA-M (at least ONE should be present):

            1. Predictive or estimation language across tenses:
               - Past-oriented: {past_tolsa_m_indicators}
               - Present-oriented: {present_tolsa_m_indicators}
               - Future-oriented: {future_tolsa_m_indicators}

            2. Temporal verb constructions across tenses:
               - Past-oriented: {past_tense_verbs}
               - Present-oriented: {present_tense_verbs}
               - Future-oriented: {future_tense_verbs}

            3. Attribution to a source making a claim about expected or forecasted outcomes
               - "according to [source]", "[source] predicts", "said [source]"

            4. Temporal markers indicating expectation timing:
               - "by 2025", "in Q3", "over the next year", "expected in 2029"
        
            Note on Indicator Types:
            - Indicators (1) = explicit predictive words like "forecast", "expects"
            - Tense verbs (2) = grammatical helpers like "will", "might", "is expected to"
        
            Note: The statement must contain both a target entity AND a measurable outcome to qualify as a TOLSA-M.
        """
        return tolsa_m_requirements

    def get_prediction_properties_and_requirements():
        """
        Convenience method to retrieve both properties and requirements together.
        Useful for comprehensive prompt construction.
        """
        return PredictionProperties.get_tolsa_m_properties(), PredictionProperties.get_requirements()

    def get_source_examples():
        """
        Returns diverse examples of source entities across multiple domains.
        Includes various combinations of name, title, organization, and identifiers.
        """
        examples = [
            "Goldman Sachs",                                                        # finance - organization
            "Goldman Sachs analyst",                                                # finance - title + organization
            "Jim Cramer, a financial analyst at CNBC",                              # finance - name + title + organization
            "Stephen A. Smith",                                                     # sports - name only
            "Shannon Sharpe, founder of the Club Shay Shay podcast",               # sports - name + title + organization
            "National Weather Service",                                             # weather - organization
            "Dr. Marshall Shepherd, a meteorologist at the University of Georgia", # weather - name + title + organization
            "the Federal Reserve",                                                  # policy - organization
            "Rep. Jasmine Crockett, a congresswoman from Texas",                   # policy - name + title + organization
            "Dr. Keith L. Black, a neurosurgeon affiliated with Cedars-Sinai",     # health - name + title + organization
            "Dr. Alexa Canady, a neurosurgeon at Children's Hospital of Michigan", # health - name + title + organization
            "trade expert Michael Brown",                                           # trade - title + name
            "FitToJesus",                                                           # misc - profile name only
            "She",                                                                  # misc - gender only
            "He",                                                                   # misc - gender only
            "Jane Doe, a senior reporter at Reuters"                                # misc - name + title + organization
        ]
        return examples

    def get_target_examples():
        """
        Returns diverse examples of target entities across multiple domains.
        Targets are the entities that the TOLSA-M is about.
        """
        examples = [
            "Apple",                                                                # finance - organization
            "Google",                                                               # finance - organization
            "Trade agreements between the US and EU",                               # trade - policy/relationship
            "C.J. Stroud, a quarterback for the Houston Texans",                   # sports - name + title + organization
            "Simone Biles, a gymnast",                                              # sports - name + title
            "the New Orleans Saints",                                               # sports - organization
            "Hurricane Milton",                                                     # weather - name only
            "the Gulf Coast",                                                       # weather - location
            "President Obama, the 44th President of the United States",            # policy - name + title
            "Vice President Kamala Harris",                                         # policy - name + title
            "the Federal Reserve",                                                  # policy - organization
            "the CDC",                                                              # health - organization
            "Dr. Lisa Cooper, a physician at Johns Hopkins Medicine",               # health - name + title + organization
            "FitToCode, a fitness influencer",                                      # misc - profile name + title
            "She",                                                                  # misc - gender only
            "He"                                                                    # misc - gender only
        ]
        return examples

    def get_date_examples():
        """
        Returns examples of date formats (both standard and non-standard).
        Dates can represent declaration time or fruition time.
        """
        examples = [
            "Wednesday, August 21, 2024",       # full date with day name
            "Wed, August 21, 2024 to 11-23-2024",  # date range
            "3 minutes",                        # relative time
            "08/21/2024 to 12.21.2024",        # numeric range with different formats
            "21/08/2024",                       # international format
            "21 August 2024",                   # European format
            "1 year from now",                  # relative future time
            "2029-07-15",                       # ISO format
            "by 2025",                          # deadline format
            "in Q3",                            # quarterly format
            "over the next decade"              # duration format
        ]
        return examples

    def get_outcome_examples():
        """
        Returns examples of outcomes categorized by type:
        - Attribute of interest (what is being measured)
        - Quantifiable metric (specific values or ranges)
        - Slope (direction of change)
        
        Covers multiple domains: finance, sports, weather, policy, health, trade, misc.
        """
        attribute_of_interest_examples = [
            "stock price",           # finance
            "team win",              # sports
            "temperature",           # weather
            "voting results",        # policy
            "heart rate",            # health
            "trade agreements",      # trade - diplomatic/economic relationship
            "number of steps"        # misc
        ]
        
        quantifiable_metric_examples = [
            "from $50 to $75",                # finance
            "from 3 wins to 10 wins",         # sports
            "from 60°F to 80°F",              # weather
            "from 40% to 60% approval",       # policy
            "from 120 to 80 bpm",             # health
            "stay same",                      # trade - maintaining status quo
            "from 8 to 3",                    # misc
            "increase by 20%"                 # misc - percentage change
        ]
        
        slope_examples = [
            "increase",              # finance - upward trend
            "decline",               # sports - downward trend
            "remain stable",         # weather - no change
            "stay same",             # trade - maintain current state
            "rise sharply",          # policy - steep upward trend
            "decrease",              # health - downward trend
            "fluctuate",             # misc - variable pattern
            "grow steadily",         # misc - consistent upward trend
            "plateau"                # misc - leveling off
        ]
        
        examples = {
            "attribute_of_interest": attribute_of_interest_examples,
            "quantifiable_metric": quantifiable_metric_examples,
            "slope": slope_examples
        }
        return examples