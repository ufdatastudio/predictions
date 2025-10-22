
class PredictionProperties:

    
    def get_prediction_properties():
        prediction_properties = """ A prediction <p> = (<p_s>, <p_t>, <p_d>, <p_o>), where it consists of the following four properties:

        1. <p_s>
            - Defined as: 
                - Source entity that states the <p>
            - Characteristics:
                - A person with either: a name only, profile name only, geneder only, domain specific title only or any combination of these.
                - An associated organization
                - Named entity: Person, organization
                - Part of speech: Noun
            - Examples:
                1. A person with a name only: Detravious
                2. A person with a profile name: FitToJesus
                3. A person with a gender only: He
                4. A person with a domain specific title: reporter, analyst, expert, top executive, senior level person, etc 
                5. A person with a combination: Detravious, a reporter
                6. An associated organization: FitTo...
                7. A combination of person with associated organization: Detravious, a reporter at FitTo...

        2. <p_t>
            - Defined as: 
                - Target entity that the <p> is about
            - Characteristics:
                - Same and <p_s>
            - Examples:
                1. A person with a name only: Jakalia
                2. A person with a profile name: FitToCode
                3. A person with a gender only: She
                4. A person with a domain specific title: reporter, analyst, expert, top executive, senior level person, etc 
                5. A person with a combination: Jamari, a senior level person
                6. An associated organization: Roeh Labs
                7. A combination of person with associated organization: A senior level person at Roeh Labs named Jamari

        3. <p_d>
            - Defined as: 
                - Date when the <p> is made
                - Date when the <p> is expected to come to fruition
            - Characteristics:
                - Forecast can range from a second to anytime in the future.
                - Could answer the question: "How far out is the <p> from today?"
                - Any format
                - Must be made before it is expected to come to fruition
                - Named entity: Date, cardinal?
                - Part of speech: ?
            - Examples:
                1. Wednesday, August 21, 2024
                2. Wed, August 21, 2024 to 11-23-2024
                3. 3 minutes
                4. 08/21/2024 to 12.21.2024
                5. 21/08/2024
                6. 21 August 2024
                7. 1 year from now

        4. <p_o>
            - Defined as:
                - Outcome
            - Characteristics:
                - Relevant or misc details such as a quantifiable metric, slope, attribute of interest
                - From Pegah at USC: Value at an instant, statistical function minimum, changes over interval, second order effect comparison, recurrent pattern
                - Named entity: ?
                - Part of speech: ?
            - Examples:
                1. remain stable
                2. increase
                3. decrease from 8 to 3
                4. stock price
                5. voting results
                6. team win
                7. number of steps

    """
        return prediction_properties