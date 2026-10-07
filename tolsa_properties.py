"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

This module defines the properties, requirements, linguistic cues, and examples
for TOLSA (Target Outcome with optionaL Source, dAte) identification.

Framework derivation and related work:

    Quantifiably Verifiable Statements (QVS)
        - Indicator <- NOTE: maps to the <target> property in TOLSA.
        - Trend     <- NOTE: maps to the <outcome> property in TOLSA.

    Structured semantic/event representations:
        - ACE (Automatic Content Extraction)
        - CAMEO (Conflict and Mediation Event Observations)
        - WEIS (World Event/Interaction Survey)
        - SRL (Semantic Role Labeling)
        - Political Event Extraction using Syntax and Semantics
        - Event Extraction Frameworks and Surveys

Linguistic cues are organized as:

    - Tense: Future (TF) <- NOTE: TF maps to tense context, NOT a standalone classification signal.

    - Keyword: 
        - Predictive (KP) <- NOTE: KP maps to proposal-oriented lexical context, NOT a standalone classification signal.
        - Modal Verb (KMV) <- NOTE: KMV maps to proposal-oriented lexical context, NOT a standalone classification signal.
        - Time Expression (KTE) <- NOTE: KTE maps to the <date> property in TOLSA.
    
    - Property mappings:
        - <target> <- NOTE: Derived primarily from QVS Indicator and event participants/entities.
        - <outcome> <- NOTE: Derived primarily from QVS Trend and measurable event outcomes.
        - <source> <- NOTE: Derived from attribution, agent, actor, speaker, and source roles
                 commonly found in ACE, CAMEO, WEIS, SRL, and political-event frameworks.
        - <date> <- NOTE: Derived from temporal expressions, temporal arguments, event timing,
                 declaration timing, and fruition timing representations.
"""


class Tolsa:
    """
    Centralized class containing all TOLSA properties, linguistic cues, and examples
    across different domains and tenses.

    Linguistic cues are organized into two categories:
        1. Tense    — past, present, or future
        2. Keywords — predictive (KP), modal verbs (KMV), temporal expressions (KTE)

    KTE maps to the <date> property and is treated as contextual information,
    NOT a standalone classification signal. Time expressions alone are
    insufficient for TOLSA classification.
    """

    def get_tolsa_definition():
        """
        Returns the core definition of TOLSA.
        Used in system identity prompts and documentation.

        Structured in layers:
            Layer 1   : acronym definition
            Layer 1.1 : acronym terms used in a sentence (broad, no expansions)
            Layer 2.1 : linguistic cues unpacked (proposal, tense)
            Layer 2.2 : properties unpacked (required, then optional)
        """
        # ------------------------------------------------------------
        # Layer 1 — acronym
        # ------------------------------------------------------------
        layer_1 = (
            "TOLSA (Target Outcome with optionaL Source dAte)"
        )

        # ------------------------------------------------------------
        # Layer 1.1 — acronym terms in a sentence, no expansions yet.
        # 'proposal' is intentionally broad: a document need not contain
        # an explicit predictive keyword or modal verb to qualify.
        # 'time of declaration' quietly introduces <date> here, paid
        # off fully in Layer 2.2.
        # ------------------------------------------------------------
        layer_1_1 = (
            "is a proposal document expressed in any tense "
            "about a target with a measurable outcome "
            "whose correctness may be unknown at the time of declaration."
        )

        # ------------------------------------------------------------
        # Layer 2.1 — linguistic cues unpacked
        # proposal → KP / KMV (non-required)
        # tense    → past, present, future (broadens future-only scope)
        # ------------------------------------------------------------
        layer_2_1 = (
            "The term 'proposal' reflects the broader lexical scope of the framework: "
            "a document need not contain an explicit predictive keyword (KP) "
            "or modal verb (KMV) to qualify, as these are non-required signals. "
            "The tense spans past, present, and future, broadening "
            "the scope beyond future-only approaches."
        )

        # ------------------------------------------------------------
        # Layer 2.2 — properties unpacked, required then optional.
        # outcome's correctness restated here to tie epistemic
        # uncertainty directly to the property definition.
        # KTE resolved into <date> context, not a classifier.
        # Every TOLSA is anchored to a real-world moment of declaration;
        # <date> is only populated when the text explicitly encodes it.
        # ------------------------------------------------------------
        layer_2_2 = (
            "A TOLSA contains two required properties: "
            "a target (entity, event, or location of interest) and a "
            "measurable outcome (attribute, metric, or slope), "
            "where this outcome's correctness may be unknown at the time of declaration. "
            "It may also contain two optional properties: "
            "a source (declaring entity), and a "
            "date (declaration and expected fruition), "
            "where keyword time expressions (KTE) are captured as date context "
            "rather than standalone classification signals. "
            "While every TOLSA is implicitly anchored to a real-world moment of "
            "declaration, the date property is only populated when the text explicitly "
            "encodes declaration or fruition timing. "
            "The absence of an extractable date does not negate the "
            "knowing/unknowing of the outcome that is central to TOLSA's definition."
        )

        tolsa_definition = f"{layer_1} {layer_1_1} {layer_2_1} {layer_2_2}"
        return tolsa_definition

    def get_required_properties():
        """
        Returns the two REQUIRED properties for TOLSA identification.
        A statement MUST contain both target and outcome to qualify as TOLSA.
        """
        required_properties = """Required Properties (MUST be present):

            1. <target>
                - Defined as:
                    - Target entity, event, or location of interest that the TOLSA is about.
                - Characteristics:
                    - A person with either: a name only, profile name only, gender only,
                      domain-specific title only, or any combination of these.
                    - May include an associated organization.
                    - An event with either: an event name only, an event type only,
                      or any combination of these.
                    - A location with either: a city, state, region, country, or
                      named geographic area, or any combination of these.

            2. <outcome>
                - Defined as:
                    - Measurable outcome of the TOLSA.
                    - Correctness may be unknown at the time of declaration.
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
        Returns the two OPTIONAL properties for TOLSA identification.
        These provide additional context but are not required for classification.

        NOTE on <date>:
            Every TOLSA is implicitly anchored to a real-world moment of
            declaration. The <date> property is only populated when the text
            explicitly encodes declaration or fruition timing. The absence of
            an extractable date does not negate the knowing/unknowing of the
            outcome that is central to TOLSA's definition.

            Temporal expressions (KTE) are captured here as date context,
            NOT used as a standalone classification signal.
        """
        optional_properties = """Optional Properties (provide context):

            1. <source>
                - Defined as:
                    - Declaring entity that conveys the TOLSA.
                - Characteristics:
                    - A person with either: a name only, profile name only, gender only,
                      domain-specific title only, or any combination of these.
                    - May include an associated organization.

            2. <date>
                - Defined as:
                    - Date when the TOLSA is declared.
                    - Date when the TOLSA is expected to come into fruition.
                - Characteristics:
                    - Can contain one or both date values based on the definitions above.
                    - May answer the question: "When was the TOLSA proposed or declared?" (Date when the TOLSA is declared.)
                    - May answer the question: "When is the TOLSA expected to come into fruition?" (Date when the TOLSA is expected to come into fruition.)
                    - Any standard or non-standard date format is acceptable.
                    - Keyword time expressions (KTE) (e.g., upcoming, next in Y years,
                    near/far, tomorrow) are captured here as date context — NOT used
                    as a standalone keyword signal for classification.
"""
        return optional_properties

    def get_tolsa_properties():
        """
        Returns the complete formal definition of all TOLSA properties.
        Combines required (target, outcome) and optional (source, date) properties.
        Ordering reflects requirement hierarchy: target and outcome first,
        then optional context.
        """
        required = Tolsa.get_required_properties()
        optional = Tolsa.get_optional_properties()

        tolsa_properties = f"""A TOLSA = (<target>, <outcome>, <source>, <date>)

        {required}

        {optional}
        """
        return tolsa_properties

    def get_requirements():
        """
        Returns the linguistic cues and requirements for identifying TOLSA,
        organized into two dimensions:

            1. Tense    — past, present, or future
            2. Keywords — predictive (KP), modal verbs (KMV), temporal expressions (KTE)

        Linguistic cue limitations:
            - Tense: Future (TF)        — future tense alone does NOT indicate a TOLSA (FP);
                                          non-future tense CAN still express a TOLSA (FN).
            - Keyword: Predictive (KP)  — a predictive keyword alone does NOT indicate a TOLSA (FP);
                                          a TOLSA CAN lack a predictive keyword (FN).
            - Keyword: Modal Verb (KMV) — a modal verb alone does NOT indicate a TOLSA (FP);
                                          a TOLSA CAN lack a modal verb (FN).
            - Keyword: Time Expr (KTE)  — a time expression alone does NOT indicate a TOLSA (FP);
                                          a TOLSA CAN lack a time expression (FN).

        TOLSA moves beyond these cues by prioritizing semantics, context, and properties.
        A statement MUST contain both a <target> AND a measurable <outcome> to qualify.
        """

        # ----------------------------------------------------------------
        # TENSE
        # spans past, present, and future — broadens future-only approaches.
        # Limitation: not all future-related text is a TOLSA, and
        # non-future tense can still express a TOLSA.
        # ----------------------------------------------------------------
        future_tense = [
            "will", "shall", "would", "going to", "might",
            "should", "could", "may", "must", "can"
        ]
        past_tense = [
            "was expected to", "had forecasted", "would have",
            "was going to", "was to", "was projected to",
            "had anticipated", "was predicted to"
        ]
        present_tense = [
            "is expected to", "is forecast to", "is projected to",
            "is anticipated to", "is predicted to"
        ]

        # ----------------------------------------------------------------
        # KEYWORDS
        # Three types: predictive (KP), modal verbs (KMV), temporal expressions (KTE).
        # None are required; TOLSA prioritizes semantics and properties.
        # ----------------------------------------------------------------

        # Keyword: Predictive (KP)
        # Synonyms of forward-looking statements: anticipate, expect, forecast, project, etc.
        # Limitation: KP alone does not indicate a TOLSA; a TOLSA can lack KP.
        predictive_keywords_past = [
            "expected", "forecasted", "previously predicted",
            "projected", "anticipated", "had estimated",
            "forecasted", "predicted earlier", "prior forecast",
            "earlier projection", "had anticipated"
        ]
        predictive_keywords_present = [
            "expects", "forecasts", "predicts", "projects",
            "estimates", "anticipates", "is forecasting",
            "is projecting", "is predicting"
        ]
        predictive_keywords_future = [
            "forecast", "projection", "estimate", "outlook",
            "expectation", "anticipation", "prognosis", "guess",
            "speculation", "forecasting", "foretelling",
            "forecasted outcome", "forecast estimate", "will predict",
            "speculates"
        ]

        # Remove duplicates across keyword lists: prioritize past > present > future
        past_kp_set    = set(predictive_keywords_past)
        present_kp_set = set(predictive_keywords_present) - past_kp_set
        future_kp_set  = set(predictive_keywords_future) - past_kp_set - present_kp_set

        predictive_keywords_past    = sorted(list(past_kp_set))
        predictive_keywords_present = sorted(list(present_kp_set))
        predictive_keywords_future  = sorted(list(future_kp_set))

        # Keyword: Modal Verb (KMV)
        # Non-main verbs: will, could, might, should, etc.
        # Limitation: KMV alone does not indicate a TOLSA; a TOLSA can lack KMV.
        modal_verbs = [
            "will", "would", "could", "might", "should",
            "shall", "may", "must", "can"
        ]

        # Keyword: Time Expression (KTE)
        # e.g., upcoming, next in Y years, near/far, tomorrow, etc.
        # Limitation: KTE alone does not indicate a TOLSA; a TOLSA can lack KTE.
        # In TOLSA, KTE is captured by the <date> property as date context,
        # NOT a standalone classification signal.
        temporal_expressions = [
            "upcoming", "soon", "tomorrow", "next year",
            "in Y years", "near-term", "far-term",
            "by 2025", "in Q3", "over the next year",
            "expected in 2029", "over the next decade",
            "short-term", "long-term"
        ]

        tolsa_requirements = f"""Linguistic Cues for TOLSA Identification:

        NOTE: No single cue below is sufficient on its own. TOLSA prioritizes
        semantics and properties (target + outcome) over keyword matching.

        ── 1. TENSE ──────────────────────────────────────────────────────────────
        Tense: Future (TF)  — future-related verb constructions (necessary but NOT sufficient)
            {future_tense}

        Tense: Past — non-future tense can still express a TOLSA
            {past_tense}

        Tense: Present
            {present_tense}

        ── 2. KEYWORDS ───────────────────────────────────────────────────────────
        Keyword: Predictive (KP) — synonyms of forward-looking statements (non-required)
            Past-oriented:    {predictive_keywords_past}
            Present-oriented: {predictive_keywords_present}
            Future-oriented:  {predictive_keywords_future}

        Keyword: Modal Verb (KMV) — non-main verbs (non-required)
            {modal_verbs}

        Keyword: Time Expression (KTE) — captured as <date> context, NOT a standalone classifier
            {temporal_expressions}

        ── 3. SOURCE ATTRIBUTION ─────────────────────────────────────────────────
        Phrases attributing a claim to a declaring entity:
            "according to [source]", "[source] predicts", "said [source]",
            "[source] forecasts", "[source] expects"

        ── CLASSIFICATION REQUIREMENT ────────────────────────────────────────────
        To qualify as TOLSA, the statement MUST contain:
            - A <target>   (entity, event, or location of interest)
            - An <outcome> (attribute, metric, or slope)
        Source and date are optional but provide valuable context.
        """
        return tolsa_requirements

    def get_prediction_properties_and_requirements():
        """
        Convenience method to retrieve both properties and requirements together.
        Useful for comprehensive prompt construction.
        """
        return Tolsa.get_tolsa_properties(), Tolsa.get_requirements()

    def get_source_examples():
        """
        Returns diverse examples of source entities across multiple domains.
        Source = the declaring entity conveying the TOLSA.
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
        Target = the entity, event, or location that the TOLSA is about.
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
        Returns examples of date / temporal expression formats (standard and non-standard).

        Keyword time expressions (KTE) are NOT used as a standalone classifier.
        Instead, they are captured here as date context under the <date> property,
        representing either declaration time or fruition time.

        Temporal horizon metadata (short-term, long-term, unverifiable) can also
        be stored here to support downstream narrative tracking.
        """
        examples = [
            "Wednesday, August 21, 2024",          # full date with day name
            "Wed, August 21, 2024 to 11-23-2024",  # date range
            "3 minutes",                            # relative time
            "08/21/2024 to 12.21.2024",            # numeric range with different formats
            "21/08/2024",                           # international format
            "21 August 2024",                       # European format
            "1 year from now",                      # relative future time
            "2029-07-15",                           # ISO format
            "by 2025",                              # deadline format (KTE → date property)
            "in Q3",                                # quarterly format (KTE → date property)
            "over the next decade",                 # duration format (KTE → date property)
            "upcoming",                             # KTE → date property
            "near-term",                            # temporal horizon metadata
            "long-term",                            # temporal horizon metadata
            "unverifiable"                          # temporal horizon metadata
        ]
        return examples

    def get_outcome_examples():
        """
        Returns examples of outcomes categorized by type:
        - Attribute of interest (what is being measured)
        - Quantifiable metric (specific values or ranges)
        - Slope (direction of change / trend)

        Covers multiple domains: finance, sports, weather, policy, health, trade, misc.
        """
        attribute_of_interest_examples = [
            "stock price",           # finance
            "team win",              # sports
            "temperature",           # weather
            "voting results",        # policy
            "heart rate",            # health
            "trade agreements",      # trade
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
            "quantifiable_metric":   quantifiable_metric_examples,
            "slope":                 slope_examples
        }
        return examples


    