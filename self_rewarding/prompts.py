"""Writer and Judge prompts for the TOLSA self-rewarding data factory.

The Judge uses the additive 1-5 scoring scheme from Yuan et al. (2024,
"Self-Rewarding Language Models"), adapted to the TOLSA annotation schema and
the project's human annotation guidelines:

  - If no word/phrase fits a property, leave it blank (empty list).
  - Spans must be verbatim: exact wording, punctuation, capitalization.
  - Target = entity (person/organisation), event, or location; maps to a noun
    (part-of-speech): person, place, or thing.
  - Outcome must contain an attribute, metric, or slope.
  - Keep salutations and titles on Source spans for entity resolution.
  - Garbage in, garbage out: only confident, high-quality examples. High
    quality = at least Target and Outcome. Examples missing Target/Outcome are
    intentionally included (in a minority) so models can be evaluated on the
    "extract nothing" behaviour.
"""

TOLSA_DEFINITION = (
    "TOLSA (Target Outcome with optionaL Source and dAte) is a proposal "
    "document expressed in any tense about a target with a measurable outcome "
    "whose correctness may be unknown at the time of declaration. "
    "A document need not contain an explicit predictive keyword or modal verb "
    "to qualify. The tense spans past, present, and future. "
    "Properties: Target (what the prediction is about: an entity, event, or "
    "location, usually a noun part-of-speech), Outcome (what will happen; "
    "must contain an attribute, metric, or slope), Source (who made the "
    "prediction; optional), and Date (when; explicit or event-relative; "
    "optional)."
)

ANNOTATION_GUIDELINES = """Annotation guidelines (the same rules human annotators follow):
  - Use the exact wording from the sentence, including punctuation and capitalization.
  - If no word/phrase fits a property, leave that property's list empty.
  - Target is an entity (person/organisation), event, or location, and maps to a noun
    (part-of-speech): a person, place, or thing.
  - Outcome must contain an attribute, metric, or slope (e.g. "will rise above $90",
    "increased by 12%", "is expected to slow").
  - Keep salutations and titles in Source spans for entity resolution
    (e.g. "Dr. Smith", "President Biden").
  - Garbage in, garbage out: only confident, high-quality examples are kept.
    High quality means at least Target and Outcome are present.
  - A minority of examples intentionally lack Target and/or Outcome; they exist to
    measure the ability to NOT extract something that is not there. Their empty lists
    are correct annotations, not errors.
"""

DOMAINS = [
    "macroeconomics and central banking",
    "corporate earnings and stock markets",
    "climate, energy, and environment",
    "technology and AI industry",
    "politics and elections",
    "sports competitions",
    "public health and medicine",
    "real estate and housing",
    "commodities (oil, gold, agriculture)",
    "geopolitics and international relations",
    "consumer retail and e-commerce",
    "transportation and automotive",
]


def _format_gold_example(sentence: str, spans: dict) -> str:
    import json
    return f'Sentence: "{sentence}"\nOutput: {json.dumps(spans, ensure_ascii=False)}'


class WriterPrompt:
    """The Writer hat: generate new TOLSA Task-2 style annotated examples."""

    @staticmethod
    def system_identity() -> str:
        return (
            "You are a linguistic expert who creates high-quality training data "
            "for TOLSA (Target Outcome with optionaL Source, dAte) property "
            "extraction.\n" + TOLSA_DEFINITION
        )

    @staticmethod
    def build(n_examples: int, few_shot: list, domain: str,
              mixture: dict | None = None) -> str:
        """Build the writer user prompt.

        Parameters
        ----------
        n_examples : how many candidates to produce in this call.
        few_shot   : list of (sentence, spans_dict) gold examples.
        domain     : topical domain for this batch (diversity control).
        mixture    : optional dict overriding the default generation mixture:
                     {"full_tolsa": 0.70, "with_source": 0.15, "partial": 0.15}
        """
        import json
        mixture = mixture or {"full_tolsa": 0.70, "with_source": 0.15,
                              "partial": 0.15}
        shots = "\n\n".join(
            f"Example {i+1}:\n{_format_gold_example(s, sp)}"
            for i, (s, sp) in enumerate(few_shot)
        )
        return f"""Create {n_examples} NEW, diverse example sentences about the domain of {domain}, each with its TOLSA property annotation.

{ANNOTATION_GUIDELINES}

Schema: {{"1": [source spans], "2": [target spans], "3": [date spans], "4": [outcome spans]}}
  - "1" = source, "2" = target, "3" = date, "4" = outcome.

Generation mixture for THIS batch (approximate):
  - {mixture['full_tolsa']:.0%} full TOLSA examples: a genuine prediction/proposal with at least Target and Outcome; Date and/or Source included when natural.
  - {mixture['with_source']:.0%} full TOLSA examples that explicitly attribute the prediction to a named Source (person or organisation, with titles/salutations kept).
  - {mixture['partial']:.0%} partial examples: sentences about the domain that look prediction-like but genuinely lack a Target and/or a measurable Outcome; leave the missing properties as empty lists. These test the "extract nothing" behaviour.

Hard requirements:
  1. Every sentence must be NEW - do not copy or lightly paraphrase the examples below or any well-known sentence.
  2. Every span must be copied VERBATIM from its sentence: exact wording, punctuation, capitalization, symbols (%, $, etc.).
  3. Keep multi-word spans together as one string; each distinct span is its own list item; preserve the order of occurrence in the sentence.
  4. Do NOT join spans with "|" and do not invent, infer, or paraphrase spans.
  5. Return [] for any property not explicitly stated.
  6. Vary sentence structure, tense (past/present/future), length, and sub-topic within {domain}. Avoid repetitive templates.

Here are gold-standard examples of the required behaviour (do NOT copy them):

{shots}

Respond ONLY with a valid JSON array of {n_examples} objects, each of the form:
[{{"sentence": "...", "spans": {{"1": [...], "2": [...], "3": [...], "4": [...]}}, "is_tolsa": true}}, ...]
where "is_tolsa" is true only if the sentence is a genuine TOLSA prediction (has at least Target and Outcome).
"""


class JudgePrompt:
    """The Judge hat: additive 1-5 scoring of a candidate example.

    Mirrors Figure 2 of Yuan et al. (2024): start at 1, add one point per
    satisfied criterion; reason through each criterion before scoring.
    """

    @staticmethod
    def system_identity() -> str:
        return (
            "You are a meticulous annotation-quality judge for TOLSA "
            "(Target Outcome with optionaL Source, dAte) property extraction.\n"
            + TOLSA_DEFINITION + "\n\n" + ANNOTATION_GUIDELINES
        )

    @staticmethod
    def build(sentence: str, spans: dict) -> str:
        import json
        return f"""Review the following TOLSA example and its annotation.

Sentence: "{sentence}"
Annotation: {json.dumps(spans, ensure_ascii=False)}
(schema: "1" = source, "2" = target, "3" = date, "4" = outcome)

Score this example on an additive 5-point scale. Start with a base score of 1, then add 1 point for EACH criterion that is fully satisfied:

  1. SENTENCE QUALITY: The sentence is well-formed, realistic English, and self-contained. If it is presented as a TOLSA prediction, it genuinely is one (a proposal about a target with a measurable outcome whose correctness may be unknown at the time of declaration). If it is a partial/non-TOLSA example, it genuinely lacks the missing properties.

  2. TARGET: Every target span is an entity (person/organisation), event, or location that maps to a noun (person, place, or thing), is copied verbatim, and no explicitly stated target is missing. If there is genuinely no target, the target list is correctly empty.

  3. OUTCOME: Every outcome span contains an attribute, metric, or slope, is copied verbatim, and no explicitly stated outcome is missing. If there is genuinely no measurable outcome, the outcome list is correctly empty.

  4. SOURCE & DATE: Source spans capture who made the prediction with salutations/titles kept; date spans are explicit or event-relative time expressions; both are verbatim; empty lists are used correctly when the property is absent.

  5. SCHEMA & FORMAT: The annotation follows the schema exactly (keys "1"-"4", lists of strings), spans preserve order of occurrence, no span is invented or paraphrased, and nothing outside source/target/date/outcome was extracted.

First reason briefly through each of the 5 criteria (one or two sentences each), then give the final score as the LAST line of your response in EXACTLY this format:
Score: {{integer between 1 and 5}}
"""
