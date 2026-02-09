import re
import unicodedata
from typing import Callable, List, Literal, Set, Tuple

import lightlemma

from delta_belief_rl.utils.compiled_constants import (
    INVALID_CHARS,
    QUESTION_WORDS_LEMMATIZED,
    QUESTION_WORDS_STEMMED,
)


def normalize_text(text: str) -> str:
    """
    Normalize text by removing diacritics/accents.
    e.g., "Malé" -> "Male", "Kraków" -> "Krakow", "Nur-Sultan" -> "Nur-Sultan"
    """
    # Normalize to NFD (decomposed form), then remove combining characters (accents)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def parse_ground_truth_with_alternatives(ground_truth: str) -> List[str]:
    """
    Parse ground truth that may contain alternative names in parentheses.

    Examples:
        "Astana (Nur-Sultan)" -> ["astana", "nur-sultan"]
        "Oskemen (Ust-Kamenogorsk)" -> ["oskemen", "ust-kamenogorsk"]
        "Male, Maldives" -> ["male", "maldives"]
        "Krakow, Poland" -> ["krakow", "poland"]

    Returns a list of normalized, lowercased words that can be matched.
    """
    # Extract content inside parentheses as alternative names
    alternatives = re.findall(r"\(([^)]+)\)", ground_truth)

    # Remove parenthetical content from main text
    main_text = re.sub(r"\([^)]+\)", "", ground_truth)

    # Combine main text and alternatives
    all_text = main_text + " " + " ".join(alternatives)

    # Clean, normalize diacritics, and split into words
    cleaned = all_text.replace(",", "").strip().lower()
    normalized = normalize_text(cleaned)

    # Split on whitespace and hyphens but keep hyphenated words as well
    words = normalized.split()

    # Also add individual parts of hyphenated words
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        if "-" in word:
            expanded_words.extend(word.split("-"))

    return [w for w in expanded_words if w]  # Filter empty strings


JUDGE_METRICS_KEYS = {
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "wrong_negatives",
    "unrecognized",
}

JUDGE_EXTRAS_KEYS = {
    "possible_overfit",
    "repeated",
    "multiple_questions",
    "no_question",
    "no_question_word",
    "invalid_chars",
    "too_long_word",
    "too_long_action",
    "too_short_action",
    "invalid_by_judge",
}


def _regex_method(action: str, ground_truth: str) -> bool:
    return re.search(ground_truth, action, re.IGNORECASE) is not None


def _exact_method(split_action: List[str], ground_truth: str) -> bool:
    return ground_truth in split_action


def _syntax_method(
    action: str, ground_truth: str, complete_str: str
) -> Tuple[str | None, Set[str]]:
    """
    Employs lemmatization and stemming to compare the action and ground truth.

    Args:
        action (str): The action taken by the actor (i.e., a question or guess).
        ground_truth (str): The ground truth to verify against.
        complete_str (Str): The string indicating a completed action.

    Returns: a tuple containing:
        - str | None: the judgement of the action.
            - "finished" if the action matches the ground truth.
            - "invalid" if it contains a question word or auxiliary verb.
            - `None` otherwise (i.e., a valid question that did not ask for the ground truth).
        - Set[str]: A set of extra categories that the action was classified as.
    """

    # Remove question mark for analysis
    words = lightlemma.tokenize(action)

    secret_lemma = lightlemma.lemmatize(ground_truth)
    secret_stem = lightlemma.stem(ground_truth)

    # Check if first word is a question word or auxiliary verb
    question_word_check = False
    extras: set[str] = set()
    for word in words:
        if (
            len(word) > 45
        ):  # longest word in the English language: "pneumonoultramicroscopicsilicovolcanoconiosis"
            extras.add("too_long_word")
            continue  # do not lemmatize/stem this token

        lemma = lightlemma.lemmatize(word)
        stemmed = lightlemma.stem(word)

        if not question_word_check and (
            lemma in QUESTION_WORDS_LEMMATIZED or stemmed in QUESTION_WORDS_STEMMED
        ):
            question_word_check = True

        if lemma == secret_lemma or stemmed == secret_stem:
            return complete_str, extras

    if not question_word_check:
        extras.add("no_question_word")
        return "invalid", extras

    return None, extras


def correct_obs(
    action: str,
    obs: str,
    ground_truth: str,
    methods: Set[
        Literal[
            "exact",
            "regex",
            "syntax",
            "sentence",
            "question",
            "multiple_questions",
            "length",
        ]
    ] = {"exact", "syntax", "sentence", "question", "multiple_questions", "length"},
    false_positive_behavior: str | None = "yes",
    short_circuit: bool = False,
    debug: bool = False,
    env: str = "twenty_questions",
) -> Tuple[str, set[str]]:
    """
    Verify the observation produced by the judge and substitute observations that should be "finished".

    If the questioner has guessed the correct word (ground truth), then the observation
    should be "finished".

    Args:
        action (str): The action taken by the actor (i.e., a question or guess).
        obs (str): The observation produced by the judge.
        ground_truths (str): The ground truth to verify against.
        method (Set[Literal["regex", "syntax"]]): The set of methods to use for verification;
          any method yielding a positive result will make the observation "finished".
        - "exact": Check if the ground truth is present exactly as a word in the action.
        - "regex": Use a regex to check if the ground truth is in the action.
        - "syntax": Compare the lemmas & roots of the action and ground truth.
        - "sentence": Check if action is a single sentence based on break and whitespace tokens.
        - "question": Mark the action as "invalid" if it does end with a question mark.
        - "multiple_questions": Mark the action as "invalid" if it contains multiple question marks.
        - "length": Check if the action is within reasonable length bounds (2-500 characters).
        false_positive_behavior (str | None): The string to change the observation to if
          a false positive is detected.
        - If `None`, the observation will not be changed.
        short_circuit (bool): Whether to short circuit the checks whenever possible (`True`) or to
          perform all checks to obtain more accurate metrics (`False`).
        debug (bool): Whether to print debugging statements.

    Returns:
        Tuple[str, set[str]]: A tuple containing:
        - The corrected observation
        - The set of extra categories that the observation was classified as; these are informative
          cases that do not change the observation, but are useful for debugging or analysis.
    """

    if env == "twenty_questions":
        completed_str = "finished"
    elif env == "guess_my_city" or env == "customer_service":
        completed_str = "goal reached"  # lowercase to match obs.lower()
    else:
        raise ValueError(f"Unknown environment: {env}")

    action = action.strip().lower()
    obs = obs.strip().lower()
    ground_truth = ground_truth.strip().lower()

    invalid = False
    extras: set[str] = set()

    ## Validity detection methdods: make it "invalid" if the action does not meet the criteria

    if "length" in methods:
        if len(action) > 500:
            if debug:
                print(f"[DEBUG] Action '{action}' is too long.")
            extras.add("too_long_action")
            if short_circuit:
                return "invalid", extras
            invalid = True
        elif len(action) < 2:
            if debug:
                print(f"[DEBUG] Action '{action}' is too short.")
            extras.add("too_short_action")
            if short_circuit:
                return "invalid", extras
            invalid = True

    if "question" in methods:
        if not action.endswith("?"):
            if debug:
                print(f"[DEBUG] Action '{action}' is not a question.")
            extras.add("no_question")
            if short_circuit:
                return "invalid", extras
            invalid = True

    if "multiple_questions" in methods:
        count_questions = action.count("?")
        if count_questions > 1:
            if debug:
                print(f"[DEBUG] Action '{action}' contains multiple question marks.")
            extras.add("multiple_questions")
            if short_circuit:
                return "invalid", extras
            invalid = True

    if "sentence" in methods:
        if bool(INVALID_CHARS.search(action)):
            if debug:
                print(f"[DEBUG] Detected invalid characters in action '{action}'")
            extras.add("invalid_chars")
            if short_circuit:
                return "invalid", extras
            invalid = True

    syntax_result = None
    if "syntax" in methods:
        syntax_result, syntax_extras = _syntax_method(
            action, ground_truth, completed_str
        )
        extras.update(syntax_extras)
        if syntax_result == "invalid":
            if debug:
                print(f"[DEBUG] Syntax check caught invalid action '{action}'")
            if short_circuit:
                return "invalid", extras
            invalid = True

    ## False negatives detection methods: make it positive (i.e., "finished"), unless the judge deemed it "invalid"

    # Conditions should be ordered in increasing fashion by their expected time complexity to optimize short-circuit evaluation
    pos_conds: List[Callable[[], bool]] = []
    if env == "twenty_questions":
        if "syntax" in methods:
            pos_conds.append(lambda: syntax_result == completed_str)
        if "regex" in methods:
            pos_conds.append(lambda: _regex_method(action, ground_truth))
        if "exact" in methods:
            cleaned_action = re.sub(r"[^\w\s-]|[\d]", "", action)
            words = cleaned_action.split()
            parsed_words = [word.strip().lower() for word in words]
            pos_conds.append(lambda: _exact_method(parsed_words, ground_truth))

    elif env == "guess_my_city":
        # split the ground truth into words, as it contains "city, country"
        ground_truth_words = ground_truth.replace(",", "").lower().split()
        if "exact" in methods:
            # exact matching: all ground truth words must be present in the action
            cleaned_action = re.sub(r"[^\w\s-]|[\d]", "", action)
            words = cleaned_action.split()
            parsed_words = [word.strip().lower() for word in words]
            pos_conds.append(
                lambda pw=parsed_words, gtw=ground_truth_words: all(
                    w in pw for w in gtw
                )
            )
        if "regex" in methods:
            # regex matching: all ground truth words must be present (AND condition, order independent)
            pos_conds.append(
                lambda act=action, gtw=ground_truth_words: all(
                    re.search(r"\b" + re.escape(w) + r"\b", act, re.IGNORECASE)
                    is not None
                    for w in gtw
                )
            )
    elif env == "customer_service":
        if "syntax" in methods or "regex" in methods or "exact" in methods:
            print(
                "Warning: 'syntax', 'regex', and 'exact' methods are not applicable in 'customer_service' environment."
            )

    if any(cond() for cond in pos_conds):  # short-circuit evaluation enabled by `any`
        if invalid:  # we did not short-circuit previously
            return "invalid", extras

        return completed_str, extras

    elif invalid:  # we did not short-circuit previously
        return "invalid", extras

    # The judge caught an invalid action that we didn't
    elif obs == "invalid":
        extras.add("invalid_by_judge")
        return obs, extras

    # Repeated detection (performed by the judge, so only reporting)
    elif obs == "repeated":
        extras.add("repeated")
        return obs, extras

    ## False positives detection methods: change the observation to the specified behavior
    elif completed_str in obs and false_positive_behavior is not None:
        return false_positive_behavior, extras

    return obs, extras


def correct_obs_gmc(
    action: str,
    obs: str,
    ground_truth: str,
    methods: Set[
        Literal[
            "exact",
            "multiple_bets",
            "multiple_questions",
        ]
    ] = {"multiple_bets", "exact", "multiple_questions"},
    false_positive_behavior: str | None = "notvalid",
    debug: bool = False,
    env: str = "guess_my_city",
) -> Tuple[str, set[str]]:
    """
    Validate the judge observation against the action and ground truth, correcting false positives.

    This function should only be called when the judge has said "goal reached" (done=True).
    It checks:
    1. No multiple question marks in action → invalid_mq
    2. No multiple bets (action contains "or" or 2+ commas) → notvalid
    3. At least one ground truth word is present in the action → if not, false positive

    Args:
        action (str): The action taken by the actor (question or guess).
        obs (str): The observation produced by the judge.
        ground_truth (str): The ground truth to verify against (e.g., "Paris, France").
        methods (Set[Literal["exact", "multiple_bets", "multiple_questions"]]):
            Verification methods to apply.
            - "multiple_questions": Check for multiple question marks (invalid_mq if found).
            - "multiple_bets": Check for "or" keyword or 2+ commas (notvalid if found).
            - "exact": Check if at least one gt word is in the action.
        false_positive_behavior (str | None): Replacement when judge said "goal reached" but
            no ground truth word is found in the action.
        debug (bool): Whether to print debug messages.
        env (str): Environment name (unused, kept for API compatibility).

    Returns:
        Tuple[str, set[str]]: The corrected observation and a set of diagnostic tags.
    """

    action = action.strip().lower()
    obs = obs.strip().lower()
    ground_truth = ground_truth.strip().lower()

    extras: set[str] = set()

    if "multiple_questions" in methods:
        if action.count("?") > 1:
            if debug:
                print(f"[DEBUG] Action '{action}' contains multiple question marks.")
            extras.add("multiple_questions")
            return "invalid_mq", extras

    if "multiple_bets" in methods:
        if re.search(r"\bor\b", action):
            if debug:
                print(f"[DEBUG] Action '{action}' contains multiple options ('or').")
            extras.add("multiple_bets")
            return "invalid_mb", extras

    if "exact" in methods:
        # Parse ground truth with alternative names (handles parentheses) and normalize diacritics
        ground_truth_words = parse_ground_truth_with_alternatives(ground_truth)

        # Clean and normalize the action (remove diacritics like é -> e, ó -> o)
        cleaned_action = re.sub(r"[^\w\s-]|[\d]", "", action)
        normalized_action = normalize_text(cleaned_action)
        parsed_words = normalized_action.split()

        # Also expand hyphenated words in the action
        expanded_action_words = []
        for word in parsed_words:
            expanded_action_words.append(word)
            if "-" in word:
                expanded_action_words.extend(word.split("-"))

        # A single word match is sufficient (city name OR alternative name)
        gt_found = any(w in expanded_action_words for w in ground_truth_words)

        if not gt_found:
            if debug:
                print(
                    f"[DEBUG] No ground truth word found in action. GT words: {ground_truth_words}, Action words: {expanded_action_words}"
                )
            extras.add("gt_not_found")
            return false_positive_behavior if false_positive_behavior else obs, extras

    return obs, extras
