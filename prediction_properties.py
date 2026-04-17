class PredictionProperties:
    
    def get_prediction_properties():
        prediction_properties = """A prediction <p> = (<p_s>, <p_t>, <p_d>, <p_o>), where it consists of the following four properties:
            1. <p_s>
                - Defined as: 
                    - Source entity that conveys the <p>.
                - Characteristics:
                    - A person with either: a name only, profile name only, gender only, domain specific title only or any combination of these.
                    - An associated organization.
            2. <p_t>
                - Defined as: 
                    - Target entity that the <p> is about.
                - Characteristics:
                    - A person with either: a name only, profile name only, gender only, domain specific title only or any combination of these.
                    - An associated organization.
            3. <p_d>
                - Defined as: 
                    - Date when the <p> is made.
                    - Date when the <p> is expected to come to fruition.
                - Characteristics:
                    - Can have two different values based on how it's defined above.
                    - Could answer the question: "How far out is the <p> from today?"
                    - Any format, so follow any standard.
            4. <p_o>
                - Defined as:
                    - Outcome of the <p>.
                - Characteristics:
                    - Is comprised of one or a combination of the following: an attribute of interest, a quantifiable metric, slope.
                    - Could also be: value at an instant, statistical function minimum, changes over interval, second order effect comparison, recurrent pattern.
        """
        return prediction_properties

    def get_requirements():
        future_tense_verbs = [
            "will", "shall", "would", "going", "might",
            "should", "could", "may", "must", "can"
        ]
        synonyms = [
            "forecast", "projection", "estimate", "outlook", "expectation",
            "anticipation", "prophecy", "prognosis", "guess", "speculation",
            "forecasting", "foretelling", "forecasted outcome", "forecast estimate"
        ]
        prediction_requirements = f"""Requirements of a prediction: 
            1. Usage of synonyms to the word "prediction", such as {synonyms}.
            2. May us the future verb tense, such as: {future_tense_verbs}.
            3. Could also be a "past prediction" too.
        """
        return prediction_requirements
    
    def get_prediction_properties_and_requirements():
        return PredictionProperties.get_prediction_properties(), PredictionProperties.get_requirements()

    def get_source_examples():
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
            "FitToJesus",                                                           # misc - profile name only
            "She",                                                                  # misc - gender only
            "He",                                                                   # misc - gender only
            "Jane Doe, a senior reporter at Reuters"                                # misc - name + title + organization
        ]
        return examples

    def get_target_examples():
        examples = [
            "Apple",                                                                # finance - organization
            "Google",                                                               # finance - organization
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
        examples = [
            "Wednesday, August 21, 2024",
            "Wed, August 21, 2024 to 11-23-2024",
            "3 minutes",
            "08/21/2024 to 12.21.2024",
            "21/08/2024",
            "21 August 2024",
            "1 year from now"
        ]
        return examples

    def get_outcome_examples():
        attribute_of_interest_examples = [
            "stock price",       # finance
            "team win",          # sports
            "temperature",       # weather
            "voting results",    # policy
            "heart rate",        # health
            "number of steps"    # misc
        ]
        quantifiable_metric_examples = [
            "from $50 to $75",           # finance
            "from 3 wins to 10 wins",    # sports
            "from 60°F to 80°F",         # weather
            "from 40% to 60% approval",  # policy
            "from 120 to 80 bpm",        # health
            "from 8 to 3",               # misc
        ]
        slope_examples = [
            "increase",          # finance
            "decline",           # sports
            "remain stable",     # weather
            "rise sharply",      # policy
            "decrease",          # health
            "fluctuate",         # misc
        ]
        examples = {
            "attribute_of_interest": attribute_of_interest_examples,
            "quantifiable_metric": quantifiable_metric_examples,
            "slope": slope_examples
        }
        return examples