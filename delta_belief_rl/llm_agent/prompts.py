from typing import List


############### TWENTY QUESTIONS PROMPTS ##################

INVALID_ANSWER = (
    """You have not asked a valid question. Ask a brief yes/no question instead."""
)
REPEATED_ANSWER = """You have already asked this question before. Ask a new yes/no question instead."""

SYSTEM_PROMPT_ADJ = """You are the Questioner in the game '20 Questions'. Your objective is to guess the secret word (a common English noun) by asking a sequence of up to 20 yes-or-no questions."""

SYSTEM_PROMPT_ORIGINAL = """You are the Questioner in a game of 20 Questions, and your goal is to determine the secret word.
The secret is randomly drawn from the most frequent nouns of the English language.

Ask clear, concise, and strategic yes/no questions that will help you narrow down the possibilities.
Consider previous answers to inform your subsequent questions, and keep track of the information you gather.
Focus on deductive reasoning, start with a broad question and refine your queries as you progress."""

THINKING_PROMPT = """Ask a question to gain additional information or make an informed guess based on what you learned so far.

Instructions:
1. Provide your chain-of-thought reasoning step by step within XML <think>[Thoughts]</think> tags.
2. Don't think longer than two brief sentences.
3. Ask a question that can be answered with "Yes" or "No" to help you deduce the secret word. 
4. Do not ask multiple questions at once.
5. The same question will always result in the same answer. Do not repeat questions that have already been asked, and do not insist on guessing a word that you have already tried before.
6. If you are ready to guess the secret word, ask it directly as a question.
7. Provide your question within XML <question>[Question]</question> tags.

Ask your question or make your guess in the format <think>...</think><question>[Question]</question>: """

COT_PROMPT = """Ask a question to gain additional information or make an informed guess based on what you learned so far.

Instructions:
1. Provide your chain-of-thought reasoning step by step.
2. Don't think longer than two brief sentences.
3. Ask a question that can be answered with "Yes" or "No" to help you deduce the secret word. 
4. Do not ask multiple questions at once.
5. The same question will always result in the same answer. Do not repeat questions that have already been asked, and do not insist on guessing a word that you have already tried before.
6. If you are ready to guess the secret word, ask it directly as a question.
7. Provide your question within XML <question>[Question]</question> tags.

Please reason step by step, and ask your question or make your guess in the format <question>[Question]</question>: """

DIRECT_PROMPT = """Ask a question to gain additional information about the secret or guess what the secret is.

Instructions:
1. Ask a question that can be answered with "Yes" or "No" to help you deduce the secret word.
2. Your answer must be a single question. Do not provide any additional commentary or reasoning.

Ask your question: """

# This prompt requires the chat history to be given in-context for the model to be able
# to evaluate wheter the question is repeated or not.
JUDGE_SYSTEM_PROMPT_REPEATED = """You are the Answerer in a game of 20 Questions. 
Your role is to answer the questioner's questions with short and precise responses: 'Yes','No', 'Invalid', 'Repeated', or 'Finished'. 
Do not provide extra hints or commentary beyond your answer. Remain consistent and ensure that your responses accurately reflect the secret word. Answer the question to the best of your abilities."""

JUDGE_SYSTEM_PROMPT = """You are the Answerer in a game of 20 Questions. 
Your role is to answer the questioner's questions with short and precise responses: 'Yes','No', 'Invalid', or 'Finished'. 
Do not provide extra hints or commentary beyond your answer. Remain consistent and ensure that your responses accurately reflect the secret word. Answer the question to the best of your abilities.
IMPORTANT: Always respond in English only."""


################## GUESS MY CITY PROMPTS ##################
# from guess_my_city config/paprika/guess_my_city.json, slightly adapted
JUDGE_SYSTEM_PROMPT_GMC = """You are the environment in a game called 'Guess My City.' You will be given a specific home city (e.g., London, United Kingdom) and you have to respond to the player's questions to help them guess this city.
Follow these rules:
1. Answer the agent player's questions honestly and correctly (i.e., the answers need to be true given the home city given to you at the start of the game), but do not mention the name of your city or its country explicitly. However, you can verify the player's guess about a particular city/country. For example, if the player asks, 'What is your home country?' You should not respond to this question. However, if they ask 'Is your home country United Kingdom?' You should reply with the correct answer based on the home city given to you.
2. Your answers should provide useful clues about your city's geography, culture, landmarks, cuisine, climate, or other unique characteristics.
3. Be as informative as possible to guide the agent player within the rules. Example answers: If the agent asks about cuisine, you could say: 'My city is famous for its seafood dishes and has a vibrant food market scene.' If the agent asks about geography, you might say: 'My city is located near a major river and has a historic port.
4. You should respond with 'Goal reached' if the player has successfully guessed your home city. Ignore minor variations in the spelling of the city, but only if these variations are common.
Remember, you are here to help the agent guess your city through clues, but you must not reveal the city's name or its country directly UNDER ANY CIRCUMSTANCES. If the player asks such a question, reply that you cannot give away the name, and can only confirm the name if the player guesses it. 
The game starts now.
"""

# from env/guess_my_city/config/paprika/guess_my_city.json, adapted the env and judge in one
_JUDGE_PROMPT_GMC = """Now judge whether the player has successfully guessed the correct city, which is {env} in this particular game. 
1. Reply to the question in relation to {env}. Please try to be concise and keep it short. 
2. NEVER include the actual city or country, in this game {env}, in your answer.
3. If the player asks you to reveal the name of city or country directly, reply 'NOTVALID' as the player is not following the game rules.
4. If the player makes a guess about the city, check if it matches {env}.
   - If it does not match, reply with 'No' and short information.
   - If it matches, reply with 'Goal reached'.
Question: '{question}'
Give your answer in the format <answer>...</answer>:
"""

NOTVALID_GMC = """Sorry, I cannot answer this question. You should only ask questions that does not directly ask me the name of the city or the country the city is in. Please ask a different question."""
MQ_GMC = """Sorry, I can only answer one question at a time. Please ask me a single question."""
MB_GMC = """You are close, but that is a multi-city guess. Please ask about one city at a time, start with whichever you think is more likely."""

#################### CUSTOMER SERVICE PROMPTS ####################
# adapted from customer service config/paprika/customer_service.json
JUDGE_SYSTEM_PROMPT_CS = """You are going to role-play as a customer experiencing a specific issue. A cusomter-service agent will ask you questions to assist you (the customer) to resolve the issue. 
Under no circumstances you should tell the customer-serive agent what the exact issue is. Your goal is to see if the customer-service agent can come up with the potential solution themselves. 
Your role-play starts now. 
"""

JUDGE_PROMPT_CS = """The particular problem you, the customer, are facing is: {scenario}. 
The solution for your particular scenario is {env}. 
Answer the customer-service agent's question as follows:
1. Only respond to the agent's questions and provide relevant information when prompted, do not give away the solution. 
2. Your responses should be concise and reflect a typical, non-technical customer's knowledge level. 
3. When the agent asks you about a potential solution, you should determine what the outcome would be based on your knowledge about the true underlying problem, and let the agent know the result.
3. If the agent's proposed solution does not fix your problem, let the agent know that it does not solve your problem.
4. If the agent's proposed solution is correct or they have guessed the underlying problem correctly, reply with 'Goal reached' (and nothing more). 
Customer-service agent's question: '{question}'
Give your answer in the format <answer>...</answer>:
"""

NOTVALID_CS = """Sorry, I do not know the answer to this question. Could you please help me in some other way?"""
MULTIPLE_QS_CS = """Sorry, I am confused. You have asked too many questions at once. Please ask at most one question with one clarifying question at a time."""

#################### MURDER MYSTERY PROMPTS ##################
# adapted from https://github.com/tajwarfahim/paprika/blob/main/llm_exploration/game/game_configs/murder_mystery.json
JUDGE_SYSTEM_PROMPT_MM = """You are an advanced AI responsible for simulating a challenging and immersive murder mystery game environment. 
You are provided with the full details of the scenario, including the characters, their backstories, motives, alibis, the sequence of events, and the true culprit. 
Your task is to respond to the actions and inquiries of the detective player in a manner that maintains the game's logic and adds complexity to the mystery. Always strive to make the game enjoyable and intriguing while revealing the absolute minimum information required for the detective to progress logically.
The game begins now.
"""

JUDGE_PROMPT_MM = """The true crime scenario with the actual culprit in this game is: '{env}'.
The detective's last response is: '{question}'.

Respond to the detective's response as follows:
1. Never explicitly reveal who the true culprit is in your response.
2. If the detective makes a clear, direct accusation (e.g., "[Name] is the murderer/culprit/killer") AND it matches the true culprit, respond with exactly: "Goal reached".
3. If the detective accuses the wrong person, have the game acknowledge this without confirming or denying, and allow them to continue investigating.
4. If the detective inspects an item, describe its appearance and condition. Reveal clues proportional to how specific or thorough the inspection is. Avoid gratuitously revealing details.
5. If the detective interviews a character, roleplay the character authentically based on their personality and knowledge. Characters protect themselves and may misdirect. They do not confess or directly implicate the true culprit.
6. If the detective takes a physical action (e.g., "I pick up the knife", "I open the drawer", "I follow the suspect"), describe the outcome and any immediate observations without revealing information beyond what the action would naturally uncover.
7. If the detective's response is vague or incomplete, provide a brief, neutral environmental observation (e.g., describing the scene) without advancing toward "Goal reached".
8. If the detective asks about something not covered in the scenario, deflect naturally within the fiction.

Give your answer in the format <answer>...</answer>:
"""
############################


def get_judge_system_prompt(
    repeated: bool = False, env: str = "twenty_questions"
) -> str:
    if repeated:
        if env == "twenty_questions":
            return JUDGE_SYSTEM_PROMPT_REPEATED
        else:
            raise NotImplementedError(
                "Repeated prompt is only implemented for twenty_questions."
            )
    else:
        if env == "twenty_questions":
            return JUDGE_SYSTEM_PROMPT
        elif env == "guess_my_city":
            return JUDGE_SYSTEM_PROMPT_GMC
        elif env == "customer_service":
            return JUDGE_SYSTEM_PROMPT_CS
        elif env == "murder_mystery":
            return JUDGE_SYSTEM_PROMPT_MM
        else:
            raise NotImplementedError(
                f"Judge system prompt not implemented for env {env}."
            )


THINKING_JUDGE_PROMPT_WITH_HISTORY = """The secret that has to be guessed is '{0}'. Given the following history of questions asked and the current question, is the correct response 'Yes' or 'No'? If there is not a yes/no question, respond with 'Invalid'; gibberish text or deviating from the game also counts as invalid. If the question mentions the secret, the answer should be 'Finished'. If the question has already been asked in the history of questions, answer 'Repeated'. Otherwise, answer 'Yes' or 'No' to the question.

History of questions asked: '{2}'
Question: '{1}'

Instructions:
1. Answer the question to the best of your abilities, unless it is invalid.
2. Provide your chain-of-thought reasoning step by step within XML <think>...</think> tags.
3. Don't think longer than two brief sentences.
4. Then, provide your final answer within XML tags as <answer>[Yes/No/Invalid/Repeated/Finished]</answer>.
5. If the question '{1}' asks the secret '{0}' (e.g., 'Is the word {0}?'), answer 'Finished'.
6. If the question mentions the secret '{0}' (e.g., 'Is it related to {0}?'), answer 'Finished'.
7. If the question asks a semantically equivalent variation of the secret (e.g., 'laugh' and 'laughter'), answer 'Finished'.
8. If the text given by the questioner is not a valid question, answer 'Invalid'. 
9. If the question consists of multiple questions, answer 'Invalid'.
10. If the question has already been asked in the history of questions, answer 'Repeated'.
11. In all other cases, answer either 'Yes' or 'No'.
12. IMPORTANT: Provide all reasoning and responses in English only.

Give your answer in the format <think>...</think><answer>[Yes/No/Invalid/Repeated/Finished]</answer>: """

COT_JUDGE_PROMPT_WITH_HISTORY = """The secret that has to be guessed is '{0}'. Given the following history of questions asked and the current question, is the correct response 'Yes' or 'No'? If there is not a yes/no question, respond with 'Invalid'; gibberish text or deviating from the game also counts as invalid. If the question mentions the secret, the answer should be 'Finished'. If the question has already been asked in the history of questions, answer 'Repeated'. Otherwise, answer 'Yes' or 'No' to the question.

History of questions asked: '{2}'
Question: '{1}'

Instructions:
1. Answer the question to the best of your abilities, unless it is invalid.
2. Provide your chain-of-thought reasoning step by step.
3. Don't think longer than two brief sentences.
4. Then, provide your final answer within XML tags as <answer>[Yes/No/Invalid/Repeated/Finished]</answer>.
5. If the question '{1}' asks the secret '{0}' (e.g., 'Is the word {0}?'), answer 'Finished'.
6. If the question mentions the secret '{0}' (e.g., 'Is it related to {0}?'), answer 'Finished'.
7. If the question asks a semantically equivalent variation of the secret (e.g., 'laugh' and 'laughter'), answer 'Finished'.
8. If the text given by the questioner is not a valid question, answer 'Invalid'. 
9. If the question consists of multiple questions, answer 'Invalid'.
10. If the question has already been asked in the history of questions, answer 'Repeated'.
11. In all other cases, answer either 'Yes' or 'No'.
12. IMPORTANT: Provide all reasoning and responses in English only.

Please reason step by step, and give your final answer in the format <answer>[Yes/No/Invalid/Repeated/Finished]</answer>: """


THINKING_JUDGE_PROMPT = """The secret that has to be guessed is '{0}'. Given the following question, is the correct response 'Yes' or 'No'? If there is not a yes/no question, respond with 'Invalid'; gibberish text or deviating from the game also counts as invalid. If the question mentions the secret, the answer should be 'Finished'. Otherwise, answer 'Yes' or 'No' to the question.

Question: '{1}'

Instructions:
1. Answer the question to the best of your abilities, unless it is invalid.
2. Provide your chain-of-thought reasoning step by step within XML <think>...</think> tags.
3. Don't think longer than two brief sentences.
4. Then, provide your final answer within XML tags as <answer>[Yes/No/Invalid/Finished]</answer>.
5. If the question '{1}' asks the secret '{0}' (e.g., 'Is the word {0}?'), answer 'Finished'.
6. If the question mentions the secret '{0}' (e.g., 'Is it related to {0}?'), answer 'Finished'.
7. If the question asks a semantically equivalent variation of the secret (e.g., 'laugh' and 'laughter'), answer 'Finished'.
8. If the text given by the questioner is not a valid question, answer 'Invalid'. 
9. If the question consists of multiple questions, answer 'Invalid'.
10. In all other cases, answer either 'Yes' or 'No'.
11. IMPORTANT: Provide all reasoning and responses in English only.

Give your answer in the format <think>...</think><answer>[Yes/No/Invalid/Finished]</answer>: """


COT_JUDGE_PROMPT = """The secret that has to be guessed is '{0}'. Given the following question, is the correct response 'Yes' or 'No'? If there is not a yes/no question, respond with 'Invalid'; gibberish text or deviating from the game also counts as invalid. If the question mentions the secret, the answer should be 'Finished'. Otherwise, answer 'Yes' or 'No' to the question.

Question: '{1}'

Instructions:
1. Answer the question to the best of your abilities, unless it is invalid.
2. Provide your chain-of-thought reasoning step by step.
3. Don't think longer than two brief sentences.
4. Then, provide your final answer within XML tags as <answer>[Yes/No/Invalid/Finished]</answer>.
5. If the question '{1}' asks the secret '{0}' (e.g., 'Is the word {0}?'), answer 'Finished'.
6. If the question mentions the secret '{0}' (e.g., 'Is it related to {0}?'), answer 'Finished'.
7. If the question asks a semantically equivalent variation of the secret (e.g., 'laugh' and 'laughter'), answer 'Finished'.
8. If the text given by the questioner is not a valid question, answer 'Invalid'. 
9. If the question consists of multiple questions, answer 'Invalid'.
10. In all other cases, answer either 'Yes' or 'No'.
11. IMPORTANT: Provide all reasoning and responses in English only.

Please reason step by step, and give your final answer in the format <answer>[Yes/No/Invalid/Finished]</answer>: """


JUDGE_PROMPT = """The secret that has to be guessed is '{0}'. Given the following question, is the correct response 'Yes' or 'No'? If there is not a yes/no question, respond with 'Invalid'; gibberish text or deviating from the game also counts as invalid. If the question mentions the secret, the answer should be 'Finished'. Otherwise, answer 'Yes' or 'No' to the question.

Question: '{1}'

Instructions:
1. Answer the question to the best of your abilities, unless it is invalid.
2. Then, provide your answer to be one of [Yes/No/Invalid/Finished].
3. Only give your final answer; you must not provide reasoning or commentary.
4. If the question '{1}' asks the secret '{0}' (e.g., 'Is the word {0}?'), answer 'Finished'.
5. If the question mentions the secret '{0}' (e.g., 'Is it related to {0}?'), answer 'Finished'.
6. If the question asks a semantically equivalent variation of the secret (e.g., 'laugh' and 'laughter'), answer 'Finished'.
7. If the text given by the questioner is not a valid question, answer 'Invalid'. 
8. If the question consists of multiple questions, answer 'Invalid'.
9. In all other cases, answer either 'Yes' or 'No'.
10. IMPORTANT: Provide all responses in English only.

Your answer is: """

JUDGE_PROMPT_WITH_HISTORY = """The secret that has to be guessed is '{0}'. Given the following history of questions asked and the current question, is the correct response 'Yes' or 'No'? If there is not a yes/no question, respond with 'Invalid'; gibberish text or deviating from the game also counts as invalid. If the question mentions the secret, the answer should be 'Finished'. If the question has already been asked in the history of questions, answer 'Repeated'. Otherwise, answer 'Yes' or 'No' to the question.

History of questions asked: '{2}'
Question: '{1}'

Instructions:
1. Answer the question to the best of your abilities, unless it is invalid.
2. Then, provide your answer to be one of [Yes/No/Invalid/Repeated/Finished].
3. Only give your final answer; you must not provide reasoning or commentary.
4. If the question '{1}' asks the secret '{0}' (e.g., 'Is the word {0}?'), answer 'Finished'.
5. If the question mentions the secret '{0}' (e.g., 'Is it related to {0}?'), answer 'Finished'.
6. If the question asks a semantically equivalent variation of the secret (e.g., 'laugh' and 'laughter'), answer 'Finished'.
7. If the text given by the questioner is not a valid question, answer 'Invalid'. 
8. If the question consists of multiple questions, answer 'Invalid'.
9. If the question has already been asked in the history of questions, answer 'Repeated'.
10. In all other cases, answer either 'Yes' or 'No'.
11. IMPORTANT: Provide all responses in English only.

Your answer is: """


def get_judge_prompt(
    ground_truth: str,
    question: str,
    history: List[str] | None = None,
    thinking: bool = True,
    cot: bool = False,
    env: str = "twenty_questions",
    scenario: str = None,
) -> str:
    assert not (thinking and cot), "Cannot use thinking and CoT at the same time."

    if history is None:
        if env == "twenty_questions":
            if thinking:
                return THINKING_JUDGE_PROMPT.format(ground_truth, question)
            elif cot:
                return COT_JUDGE_PROMPT.format(ground_truth, question)
            else:
                return JUDGE_PROMPT.format(ground_truth, question)
        elif env == "guess_my_city":
            return _JUDGE_PROMPT_GMC.format(env=ground_truth, question=question)
        elif env == "customer_service":
            return JUDGE_PROMPT_CS.format(
                scenario=scenario, env=ground_truth, question=question
            )
        elif env == "murder_mystery":
            return JUDGE_PROMPT_MM.format(env=ground_truth, question=question)
        else:
            raise NotImplementedError(f"Judge prompt not implemented for env {env}.")
    else:
        # Join the list of history questions into a single string, separated by "; "
        history_str = "; ".join(history)

        if thinking:
            return THINKING_JUDGE_PROMPT_WITH_HISTORY.format(
                ground_truth, question, history_str
            )
        elif cot:
            return COT_JUDGE_PROMPT_WITH_HISTORY.format(
                ground_truth, question, history_str
            )
        else:
            return JUDGE_PROMPT_WITH_HISTORY.format(ground_truth, question, history_str)


def get_question_prompt(
    response: str,
    n_questions: int,
    thinking: bool = True,
    cot: bool = False,
    max_questions: int = 20,
) -> str:
    assert not (thinking and cot), "Cannot use thinking and CoT at the same time."

    clean_response = response.strip().lower()

    # Early exits
    if clean_response == "finished":
        return f"Finished. Completed in {n_questions}/{max_questions}. Do not reply."

    # Select base prompt
    if thinking:
        prompt = THINKING_PROMPT
    elif cot:
        prompt = COT_PROMPT
    else:
        prompt = DIRECT_PROMPT

    if clean_response == "invalid" and cot:
        instruction = f"Your question was incorrectly formatted. This was question {n_questions} of {max_questions}.\n"
        return instruction + prompt
    # do not need alternative option as for the non cot case we assume that any model's ouput is a question

    # Valid response handling
    if n_questions == max_questions:
        prefix = f"{response}."
        return prefix
    else:
        prefix = f"{response}. "
        return (
            prefix + prompt
        )  # removed passing last prompt as it is not needed in the current implementation


ELICITATION_PROMPT = """Is it '{0}'?"""


def get_elicitation(gt: str) -> str:
    """
    Get the elicitation question for the given ground truth
    Args:
        gt: np.str - ground truth secret

    Returns:
        str - elicitation question
    """
    return ELICITATION_PROMPT.format(gt)
