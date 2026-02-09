import torch
import re
from typing import Set, Literal, Union, List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from delta_belief_rl.utils.format import (
    pad_dataproto_to_divisor,
    unpad_dataproto,
    nanstd,
)
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from delta_belief_rl.llm_agent.prompts import (
    get_question_prompt,
    get_elicitation,
    INVALID_ANSWER,
    REPEATED_ANSWER,
    NOTVALID_GMC,
    MQ_GMC,
    MB_GMC,
    MULTIPLE_QS_CS,
)
from delta_belief_rl.utils.syntax import (
    correct_obs,
    correct_obs_gmc,
    JUDGE_EXTRAS_KEYS,
    JUDGE_METRICS_KEYS,
)
import copy

logger = logging.getLogger(__file__)


@dataclass
class LogprobSamplingConfig:
    enabled: bool
    best_n: int
    worst_n: int
    p_best: float


@dataclass
class VerifyJudgeConfig:
    enabled: bool
    methods: Set[str]
    false_positive_behavior: str
    short_circuit: bool


@dataclass
class LogProbRewardConfig:
    enabled: bool
    base_model: Literal["actor", "ref"]
    step_model: Literal["actor", "ref"]
    agg_method: str
    normalised: bool
    methods: Set[str]
    clipping: Dict[str, Any]  # keys: enabled: bool, min: float, max: float
    tau: float
    level: Literal["trajectory", "token"]


@dataclass
class GenMultiEnvConfig:
    max_turns: int
    max_start_length: int  # orig left side length that passed as final output
    max_prompt_length: (
        int  # cut down any too long prompt (to keep in actor's vllm length)
    )
    max_obs_length: int  # cut down to judges vllm max prompt input length
    actor_cot: bool
    actor_thinking: bool
    judge_thinking: bool
    logprob_reward: LogProbRewardConfig
    logprob_sampling: LogprobSamplingConfig
    verify_judge: VerifyJudgeConfig
    repeated_prompt: bool
    debug: bool
    env: str


def _extract_question(response: str, tag: str | None) -> str:
    """
    Extract the question from the response string.

    Args:
        response: The response string containing the question.
        tag: The tag to look for (e.g., 'question'). If None, return the entire response.

    Returns:
        str: The extracted question or the entire response if tag is None.
    """

    if tag is None:
        return response.strip()

    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()  # Return only the content inside the tags
    else:
        content = ""

    return content


class LLMGenerationManager:
    def __init__(
        self,
        actor_rollout_wg,
        tokenizer_actor,  # actor tokanizer
        judge_rollout_wg,
        config: GenMultiEnvConfig,
        meta_info: Dict[str, Any] = None,
    ):
        self.actor_rollout_wg = (
            actor_rollout_wg  # should be able to access the tokenizer via this
        )
        self.tokenizer_actor = tokenizer_actor
        self.judge_rollout_wg = judge_rollout_wg  # but actually should be able to access the tokenizer via this
        self.config = config

        self.history = []
        self.meta_info = meta_info

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=self.tokenizer_actor.pad_token_id,
                end_token_id=self.tokenizer_actor.convert_tokens_to_ids("<|im_end|>"),
                start_token_id=self.tokenizer_actor.encode("<|im_start|>")[0],
                assistant_token_id=self.tokenizer_actor.encode("assistant")[0],
                new_line_token_id=self.tokenizer_actor.encode("\n")[0],
            )
        )

        print("[INFO] LLMGenerationManager initialized")
        print(self.config.logprob_reward.__dict__)
        print(self.config.logprob_sampling.__dict__)
        print(self.config.verify_judge.__dict__)

    def initialize_rollout_state(self):
        self.actor_rollout_wg.start_rollout()
        if self.judge_rollout_wg is not None:
            self.judge_rollout_wg.start_rollout()

    def shutdown_rollout_state(self):
        self.actor_rollout_wg.exit_rollout()
        if self.judge_rollout_wg is not None:
            self.judge_rollout_wg.exit_rollout()

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses.
        add_generation_prompt is hardcoded to False, as we simply process the responses_str to token_ids
        so we can format them as DataProto objects and pass them to the judge.
        Note: as add_generation_prompt is False, the value of enable_thinking is ignored (Qwen3 default is True)
        Args:
            responses: List of responses to tokenize

        Returns:
            torch.Tensor: Tokenized responses
        """

        # input here is cot + question
        chat = [[{"role": "assistant", "content": resp}] for resp in responses]
        #
        chat = self.tokenizer_actor.apply_chat_template(
            chat, add_generation_prompt=False, tokenize=False
        )
        # the output also conatins Qwen system prompt, manually remove
        chat = self._remove_system_prompt(chat)
        return self.tokenizer_actor(
            chat, add_special_tokens=False, return_tensors="pt", padding="longest"
        )["input_ids"]

    def _postprocess_responses(
        self, responses: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at question operation."""

        # should be the tokinzer for the speicifc model
        responses_str = self.tokenizer_actor.batch_decode(
            responses, skip_special_tokens=True
        )

        # only take the response until the question (in case if model generates longer)
        split_str = "</question>"  # can be ajusted to smth else
        responses_str = [
            resp.split(split_str)[0] + split_str if split_str in resp else resp
            for resp in responses_str
        ]

        responses = self._batch_tokenize(responses_str)

        return responses, responses_str

    def _remove_system_prompt(self, chat: Union[str, List[str]]) -> List[str]:
        """
        Remove any leading <|im_start|>system...<|im_end|> block from
        the prompt strings, whether chat is a single string or a list.
        """
        # Regular expression pattern to match the system prompt block (non-greedy)
        # Compile once
        system_re = re.compile(r"<\|im_start\|>system.*?<\|im_end\|>\s*", re.DOTALL)

        def strip_one(s: str) -> str:
            return system_re.sub("", s)

        if isinstance(chat, list):
            # Apply to each element
            return [strip_one(elem) for elem in chat]

        if isinstance(chat, str):
            return strip_one(chat)

        raise TypeError(
            f"_remove_system_prompt expects str or List[str], got {type(chat)}"
        )

    def _create_left_side(self, prompt) -> torch.Tensor:
        """
        Note: As add_generation_prompt is False, the value of enable_thinking is ignored (Qwen3 default is True)
        We simply encode the prompt (str) to token_ids and return the token_ids.
        Args:
            prompt: List of prompts to tokenize

        Returns:
            torch.Tensor: Tokenized system + user prompts without assistant generation tokens (i.e. <|im_end|>assistant)
        """
        chat = self.tokenizer_actor.apply_chat_template(
            prompt, add_generation_prompt=False, tokenize=False
        )
        model_inputs = self.tokenizer_actor(
            chat,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side="left",
        )
        return model_inputs.pop("input_ids")

    def _format_responses_loss(self, left_side: dict) -> Dict:
        """
        Produces:
            responses:  [B, L] int64  (right-side tokens, padded/truncated)
            loss_mask:  [B, L] bool   (True where we want to compute loss)
        Notes:
            - We assume your chat template uses <|im_start|> role ... <|im_end|> delimiters (Qwen-style).
            - We mask tokens strictly *inside* assistant segments: (<|im_start|>, 'assistant') ... <|im_end|>.
            - We exclude the 3 tokens at the segment head: <|im_start|>, 'assistant', and the newline after role (your +3 offset).
        """
        assert isinstance(self.history, list) and len(self.history) == left_side[
            "input_ids"
        ].size(0), "history list length must match batch size of left_side"

        prompt_ids = left_side["input_ids"]  # system + user instructions

        # Find all start indices where header_tokens occurs in seq
        def find_subseq_positions(seq_1d, pattern_1d):
            L, P = seq_1d.numel(), pattern_1d.numel()
            if P == 0 or P > L:
                return torch.empty(0, dtype=torch.long, device=seq_1d.device)
            windows = seq_1d.unfold(0, P, 1)  # [L-P+1, P] view
            matches = (windows == pattern_1d).all(dim=1)
            return torch.nonzero(matches, as_tuple=False).flatten()

        im_start_id = self.tensor_fn.config.start_token_id
        im_end_id = self.tensor_fn.config.end_token_id
        assert im_start_id is not None and im_start_id != -100, (
            "Missing <|im_start|> token id"
        )
        assert im_end_id is not None and im_end_id != -100, (
            "Missing <|im_end|> token id"
        )
        assistant_header = self.tokenizer_actor.encode(
            "assistant\n", add_special_tokens=False
        )  # [77091, 198]
        header_tokens = torch.tensor([im_start_id, *assistant_header])

        responses_list = []
        masks_list = []

        for idx, hist in enumerate(self.history):
            # select history that is not empty
            valid_hist = [h for h in hist if isinstance(h, dict)]
            # need to keep the system prompt as otherwise get the default one
            chat = self.tokenizer_actor.apply_chat_template(
                valid_hist, add_generation_prompt=False, tokenize=False
            )
            model_inputs = self.tokenizer_actor(
                chat, add_special_tokens=False, return_tensors="pt", padding=False
            )
            # select the prompt id
            # select non padding tokens of the prompt
            active_ids = int(
                (prompt_ids[idx] != self.tokenizer_actor.pad_token_id).sum().item()
            )
            # slice off the system + user instructions
            input_ids = model_inputs.pop("input_ids")[
                :, active_ids:
            ]  # 1, seq_len_response_only
            attention_mask = model_inputs.pop("attention_mask")[:, active_ids:]

            # build loss mask on the response tokens, find segmenst of assistant tokens
            seq = input_ids[0]  # seq_len_response_only
            loss_mask_seq = torch.zeros_like(seq, dtype=torch.bool)

            start_pos = find_subseq_positions(seq, header_tokens)
            end_pos = torch.nonzero(seq == im_end_id, as_tuple=False).flatten()

            # For each start, pair with the next end after it
            # and set mask to True over (s+3 ... e-1)  (skip <im_start>, 'assistant', '\n')
            for id_start, s in enumerate(start_pos.tolist()):
                start_inclusive = s + header_tokens.numel()
                e_candidates = end_pos[end_pos > start_inclusive]
                if e_candidates.numel() == 0:
                    continue
                e = int(e_candidates[0].item())
                if e > start_inclusive:
                    loss_mask_seq[start_inclusive:e] = True

            # Truncate from the left if we already exceed the configured max length
            cur_len = input_ids.shape[1]
            if cur_len > self.config.max_prompt_length and self.config.env in [
                "customer_service",
                "murder_mystery",
            ]:
                print(
                    f"[WARN] responses+history length {cur_len} exceeds max_prompt_length {self.config.max_prompt_length}; "
                    "left-truncating before padding"
                )
                input_ids = input_ids[:, -self.config.max_prompt_length :]
                attention_mask = attention_mask[:, -self.config.max_prompt_length :]
                loss_mask_seq = loss_mask_seq[-self.config.max_prompt_length :]

            # now pad to match the same length
            pad_len = self.config.max_prompt_length - input_ids.shape[1]
            # apply left padding
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_prompt_length,
                pad_token_id=self.tokenizer_actor.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            pad_loss = loss_mask_seq.new_zeros(pad_len, dtype=torch.bool)
            loss_mask_seq = torch.cat([pad_loss, loss_mask_seq], dim=0)

            responses_list.append(input_ids)  # [1, L_max]
            masks_list.append(loss_mask_seq.unsqueeze(0))  # [1, L_max]

            if self.config.debug and idx == 0:
                print("[DEBUG] IDs:", input_ids[0, -50:])
                print("[DEBUG] Mask:", loss_mask_seq[-50:])
                print(
                    "[DEBUG] Masked text:",
                    self.tokenizer_actor.decode(input_ids[0][loss_mask_seq]),
                )

        # Stack
        responses = torch.cat(responses_list, dim=0)
        loss_mask = torch.cat(masks_list, dim=0)

        # sanity check
        assert loss_mask.shape[0] == responses.shape[0], (
            f"loss_mask and responses must have the same batch size, got {loss_mask.shape[0]} and {responses.shape[0]}"
        )
        assert loss_mask.shape[1] == responses.shape[1], (
            f"loss_mask and responses must have the same sequence length, got {loss_mask.shape[1]} and {responses.shape[1]}"
        )

        return {"responses": responses, "loss_mask": loss_mask}

    def _postprocess_predictions(
        self, predictions: List[str]
    ) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.

        Args:
            predictions: List of raw predictions

        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):  # for llm output
                pattern = r"<(question)>(.*?)</\1>"
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(
                        2
                    ).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ""
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents

    def _ask_question(
        self,
        responses_ids: torch.Tensor,
        responses_str: List[str],
        gts: List[str],
        history: List[dict] | None = None,
        cot: bool = False,
        thinking: bool = False,
        scenarios: List[str] | None = None,
    ) -> Tuple[
        List[str], List[int], List[int], str, Dict[str, List[int]], Dict[str, List[int]]
    ]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM

        Args:
            responses_ids: tokenized question
            responses_str: raw question
            gts: ground truths
            history: history of the game
            cot: whether to use chain-of-thought reasoning
            thinking: whether to use thinking mode
            scenarios: List[str] | None = None, for customer_service environment, the scenario description
        Returns:
            List of observation strings
            dones: List[int]: 1-finihsed, 0-active
        """

        # Prepare lists
        next_obs: List[str] = []
        dones: List[int] = []
        valid_action: List[int] = []
        judge_metrics: Dict[str, List[int]] = {key: [] for key in JUDGE_METRICS_KEYS}
        judge_extras: Dict[str, List[int]] = {key: [] for key in JUDGE_EXTRAS_KEYS}

        def _uniformly_increase_dict(dict: Dict[str, List[int]], keys: Set[str]):
            max_len = max([len(v) for v in list(dict.values())]) if dict else 0

            for key in keys:
                if key not in dict:
                    dict[key] = [0 for _ in range(max_len)]
                elif len(dict[key]) < max_len:  # should not happen
                    dict[key] += [
                        0 for _ in range(max_len - len(dict[key]))
                    ]  # not a good thing to do
                    print(
                        "[WARN] Possible bug: the metrics & extras are not being incremented uniformly"
                    )
                dict[key].append(1)

            for k in dict.keys():
                if k not in keys:
                    dict[k].append(0)

        def _increase_judge_metrics(key: str):
            _uniformly_increase_dict(judge_metrics, {key})

        def _increase_judge_extras(keys: Set[str]):
            _uniformly_increase_dict(judge_extras, keys)

        # extract from history only the questions asked
        if history is not None:
            # Extract all user questions from each history entry
            all_questions = [
                [
                    _extract_question(
                        entry["content"].replace("\n", ""),
                        self.config.actor_thinking and "question" or None,
                    )
                    for entry in h
                    if entry.get("role") == "assistant"
                ]
                for h in history
            ]
            assert len(all_questions) == len(gts), (
                f"Number of questions and ground truths must match, "
                f"got {len(all_questions)} and {len(gts)}"
            )

        if not cot and not thinking:
            cur_actions = ["question"] * responses_ids.shape[0]

            # check that inputs do not exceed judge input length
            if responses_ids.shape[1] > self.config.max_obs_length:
                print(
                    f"[WARN] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {responses_ids.shape[1]} & {self.config.max_obs_length}"
                )
                responses_ids = responses_ids[:, : self.config.max_obs_length]

            # remove <im_start>assistant and <im_end> tags
            __debug_responses_ids_sz_1 = responses_ids.shape[0]
            responses_ids = self.tensor_fn._remove_assistant_token(responses_ids)
            assert responses_ids.shape[0] == len(cur_actions), (
                f"The first dimension of responses_ids changed, expected {len(cur_actions)}, got {responses_ids.shape[0]}"
            )
            __debug_responses_ids_sz_2 = responses_ids.shape[0]

            if history is not None:
                proto = DataProto.from_dict(
                    tensors={"input_ids": responses_ids},
                    non_tensors={"golden_answers": gts},
                    # pass as meta info as non_tensors have to be numpy array
                    meta_info={"history": all_questions},
                )
            else:
                if scenarios is not None:
                    proto = DataProto.from_dict(
                        tensors={"input_ids": responses_ids},
                        non_tensors={"golden_answers": gts, "scenarios": scenarios},
                    )
                else:
                    proto = DataProto.from_dict(
                        tensors={"input_ids": responses_ids},
                        non_tensors={"golden_answers": gts},
                    )

            # Generate answers
            proto_padded, pad_size = pad_dataproto_to_divisor(
                proto, self.judge_rollout_wg.world_size
            )
            output_padded = self.judge_rollout_wg.generate_sequences(proto_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)
            answers = [
                ans.strip() for ans in output.non_tensor_batch["answers"].tolist()
            ]

            # Validate count
            assert responses_ids.shape[0] == len(cur_actions), (
                f"The first dimension of responses_ids changed, expected {len(cur_actions)}, got {responses_ids.shape[0]}"
            )
            assert len(answers) == len(cur_actions), (
                f"Expected {len(cur_actions)} answers, got {len(answers)}; responses_ids.shape={responses_ids.shape}; len(gts)={len(gts)}; __debug_responses_ids_sz_1={__debug_responses_ids_sz_1}; __debug_responses_ids_sz_2={__debug_responses_ids_sz_2}; pad_size={pad_size}; output_padded.shape={output_padded.shape}"
            )
            assert len(answers) == responses_ids.shape[0], (
                f"Expected {responses_ids.shape[0]} answers, got {len(answers)}"
            )
        else:
            # extract only valid questions from the think + question tags
            cur_actions, contents = self._postprocess_predictions(responses_str)

            # Identify indices to ask
            question_indices = [i for i, a in enumerate(cur_actions) if a == "question"]
            if question_indices:
                questions = [contents[i] for i in question_indices]
                gts_sub = [gts[i] for i in question_indices]

                tokens = self.tokenizer_actor(
                    questions,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding="longest",
                )["input_ids"]

                if tokens.shape[1] > self.config.max_obs_length:
                    print(
                        f"[WARN] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {tokens.shape[1]} & {self.config.max_obs_length}"
                    )
                    tokens = tokens[:, : self.config.max_obs_length]

                if history is not None:
                    all_questions_sub = [all_questions[i] for i in question_indices]
                    proto = DataProto.from_dict(
                        tensors={"input_ids": tokens},
                        non_tensors={"golden_answers": gts_sub},
                        meta_info={"history": all_questions_sub},
                    )
                else:
                    proto = DataProto.from_dict(
                        tensors={"input_ids": tokens},
                        non_tensors={"golden_answers": gts_sub},
                    )

                proto_padded, pad_size = pad_dataproto_to_divisor(
                    proto, self.judge_rollout_wg.world_size
                )
                output_padded = self.judge_rollout_wg.generate_sequences(proto_padded)
                output = unpad_dataproto(output_padded, pad_size=pad_size)
                answers = [
                    ans.strip() for ans in output.non_tensor_batch["answers"].tolist()
                ]

                assert len(answers) == len(question_indices), (
                    f"Expected {len(question_indices)} answers, got {len(answers)}"
                )
            else:
                # exit as no valid questions asked, nothing to process
                valid_action = [0] * len(cur_actions)
                dones = [0] * len(cur_actions)  # keep all active
                next_obs = [INVALID_ANSWER] * len(cur_actions)
                return next_obs, dones, valid_action

        # Fill outputs
        ans_ptr = 0
        for i, action in enumerate(cur_actions):
            verification_class = ""

            if action == "question":
                assert ans_ptr < len(answers), (
                    f"Expected at least {ans_ptr} answers, got {len(answers)}"
                )

                # Fallback for invalid responses
                if self.config.env == "twenty_questions":
                    ans = answers[ans_ptr].strip().lower().split()
                    ans = ans[0] if len(ans) > 0 else ""
                    if ans not in (
                        "yes",
                        "no",
                        "repeated",
                        "invalid",
                        "finished",
                    ):
                        if self.config.debug:
                            print(
                                f"[WARN] Unrecognized judge answer with index {i} for question '{responses_str[i]}' (ground truth: '{gts[i]}'): {ans}. Defaulting to 'invalid'."
                            )
                        ans = "invalid"
                        verification_class = "unrecognized"

                    done = int(ans.startswith("finished"))

                    # Verify the judge's answer
                    if self.config.verify_judge.enabled:
                        corrected_obs, extras = correct_obs(
                            action=responses_str[i],
                            obs=ans,
                            ground_truth=gts[i],
                            methods=self.config.verify_judge.methods,
                            false_positive_behavior=self.config.verify_judge.false_positive_behavior,
                            short_circuit=self.config.verify_judge.short_circuit,
                            debug=self.config.debug,
                            env="twenty_questions",
                        )
                        corrected_obs = corrected_obs.strip().lower()

                        _increase_judge_extras(extras)

                        if corrected_obs != ans:
                            if ans == "finished":
                                done = 0  # flip done
                                verification_class = "false_positives"
                                if self.config.debug:
                                    print(
                                        f"[WARN] FALSE_POS: index = {i}; question = '{responses_str[i]}'; ground truth = '{gts[i]}'; corrected_obs = '{corrected_obs}'; judge_obs = '{ans}'"
                                    )
                            elif corrected_obs == "finished":
                                done = 1  # flip done
                                if verification_class != "unrecognized":
                                    verification_class = "false_negatives"
                                if self.config.debug:
                                    print(
                                        f"[WARN] FALSE_NEG: index = {i}; question = '{responses_str[i]}'; ground truth = '{gts[i]}'; corrected_obs = '{corrected_obs}'; judge_obs = '{ans}'"
                                    )
                            elif verification_class != "unrecognized":
                                verification_class = "wrong_negatives"
                                if self.config.debug:
                                    print(
                                        f"[WARN] WRONG_NEG: index = {i}; question = '{responses_str[i]}'; ground truth = '{gts[i]}'; corrected_obs = '{corrected_obs}'; judge_obs = '{ans}'"
                                    )
                            ans = corrected_obs  # flip answer
                        elif ans == "finished":
                            verification_class = "true_positives"
                        elif verification_class != "unrecognized":
                            verification_class = "true_negatives"

                        _increase_judge_metrics(verification_class)

                        # only apply this to the ans variable (not the judge_ans)
                        if ans.strip().lower() == "repeated":
                            if self.config.debug:
                                print(
                                    f"[WARN] Repeated question at index {i} for question '{responses_str[i]}' with gt '{gts[i]}': {ans}"
                                )
                                print(f"[WARN] History: {all_questions[i]}")
                            # adjust answer
                            ans = REPEATED_ANSWER

                        if ans.strip().lower() == "invalid":
                            # adjust answer
                            ans = INVALID_ANSWER
                elif self.config.env == "guess_my_city":
                    # here potentially implement verification step on the judge
                    ans = answers[ans_ptr].strip().lower()
                    done = 1 if "goal reached" in ans else 0

                    # on any question check that only single question asked
                    if (
                        "multiple_questions" in self.config.verify_judge.methods
                        and not done
                    ):
                        corrected_obs, _ = correct_obs_gmc(
                            action=responses_str[i],
                            obs=ans,
                            ground_truth=gts[i],
                            methods={"multiple_questions"},
                            false_positive_behavior="notvalid",
                            debug=self.config.debug,
                            env="guess_my_city",
                        )
                        if corrected_obs != ans:
                            if self.config.debug:
                                logger.warning(
                                    f"MQ: model cheats by asking multiple questions, question asked {responses_str[i]}"
                                )
                            ans = corrected_obs

                    # only adjust judges response if it was 'goal reached
                    if self.config.verify_judge.enabled and done:
                        corrected_obs, _ = correct_obs_gmc(
                            action=responses_str[i],
                            obs=ans,
                            ground_truth=gts[i],
                            methods=self.config.verify_judge.methods,
                            false_positive_behavior="notvalid",
                            debug=self.config.debug,
                            env="guess_my_city",
                        )
                        corrected_obs = corrected_obs.strip().lower()

                        if corrected_obs != ans:
                            done = 0  # flip done
                            logger.warning(
                                f"FALSE_POS: index = {i}; question = '{responses_str[i]}'; ground truth = '{gts[i]}'; corrected_obs = '{corrected_obs}'; judge_obs = '{ans}'"
                            )
                            ans = corrected_obs  # flip answer

                    if ans == "notvalid" or ans == "invalid":
                        ans = NOTVALID_GMC
                    if ans == "invalid_mq":
                        ans = MQ_GMC
                    if ans == "invalid_mb":
                        ans = MB_GMC
                elif self.config.env == "customer_service":
                    ans = answers[ans_ptr].strip().lower()
                    done = 1 if "goal reached" in ans else 0

                    if self.config.verify_judge.enabled:
                        # For customer_service, treat judge "goal reached" as final and avoid
                        # flipping it to invalid via false_positive_behavior.
                        corrected_obs, _ = correct_obs(
                            action=responses_str[i],
                            obs=ans,
                            ground_truth=gts[i],
                            methods=self.config.verify_judge.methods,
                            false_positive_behavior=None,
                            short_circuit=self.config.verify_judge.short_circuit,
                            debug=self.config.debug,
                            env="customer_service",
                        )
                        corrected_obs = corrected_obs.strip().lower()

                        if corrected_obs != ans:
                            if "goal reached" in ans:
                                done = 0  # flip done
                                if self.config.debug:
                                    print(
                                        f"[WARN] FALSE_POS: index = {i}; question = '{responses_str[i]}'; ground truth = '{gts[i]}'; corrected_obs = '{corrected_obs}'; judge_obs = '{ans}'"
                                    )
                            elif "goal reached" in corrected_obs:
                                done = 1  # flip done
                                if self.config.debug:
                                    print(
                                        f"[WARN] FALSE_NEG: index = {i}; question = '{responses_str[i]}'; ground truth = '{gts[i]}'; corrected_obs = '{corrected_obs}'; judge_obs = '{ans}'"
                                    )
                            ans = corrected_obs  # flip answer

                    if ans == "invalid":
                        ans = MULTIPLE_QS_CS

                elif self.config.env == "murder_mystery":
                    ans = answers[ans_ptr].strip().lower()
                    done = 1 if "goal reached" in ans else 0

                else:
                    raise NotImplementedError(
                        f"Environment {self.config.env} not implemented in LLMGenerationManager._ask_question"
                    )

                if done and self.config.debug:
                    output_str = output.non_tensor_batch["responses_str"].tolist()[i]
                    print("[DEBUG] Finished trajectory")
                    print(f"[DEBUG] secret to be guessed '{gts[i]}' ")
                    print(f"[DEBUG] questioner last question asked  {responses_str[i]}")
                    print(f"[DEBUG] judge output for finished trajectory {output_str}")

                next_obs.append(ans)
                dones.append(done)
                valid_action.append(1)
                ans_ptr += 1
            else:
                next_obs.append(INVALID_ANSWER)
                dones.append(0)
                valid_action.append(0)

        done_indices = [idx for idx, d in enumerate(dones) if d == 1]
        output_str = (
            output.non_tensor_batch["responses_str"].tolist()[done_indices[0]]
            if done_indices
            else output.non_tensor_batch["responses_str"].tolist()[0]
        )

        del output_padded, output, proto_padded, proto

        return next_obs, dones, valid_action, output_str, judge_metrics, judge_extras

    def _format_input(
        self, prompt: List[dict], generation_prompt=True, tokenizer=None
    ) -> DataProto:
        """
        Note: as add_generation_prompt is True, therefore we specify the enable_thinking parameter as coming from the base_multiturn.yaml
        Note: this parameter should be matched with the data generation parameter (whether the cot or non_cot user prompt is used)
        Args:
            prompt: List of prompts to tokenize
            generation_prompt: Whether to add the generation prompt to the prompt (Default True)

        Returns:
            DataProto: DataProto object containing the tokenized prompts
        """
        if tokenizer is None:
            tokenizer = self.tokenizer_actor
        assert isinstance(prompt, list), f"prompt must be a list, got {type(prompt)}"
        chat = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=self.config.actor_thinking,
        )
        assert len(chat) == len(prompt), (
            f"chat and raw prompt must have the same length, got {len(chat)} and {len(prompt)}"
        )
        model_inputs = tokenizer(
            chat,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side="left",
        )

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if input_ids.shape[1] > self.config.max_prompt_length:
            print(
                f"[WARN] PROMPT TOO LONG, CONSIDER CHANGING YOUR CONFIG, {input_ids.shape[1]} & {self.config.max_prompt_length}"
            )
            input_ids = input_ids[:, -self.config.max_prompt_length :]
            attention_mask = attention_mask[:, -self.config.max_prompt_length :]

        position_ids = compute_position_id_with_mask(attention_mask)

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_single_dict(data)
        data.meta_info = self.meta_info
        return data

    def _create_responses_mask(
        self, prompts: list[list[dict]], secrets, tok
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        response_tokens: (B, T) left-padded tokens of the last [user, assistant] pair
        mask           : (B, T) left-padded 0/1 mask for the secret span within that pair
        """

        pairs = [p[-2:] for p in prompts]

        texts = tok.apply_chat_template(
            pairs,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=getattr(self.config, "actor_thinking", False),
        )
        texts = self._remove_system_prompt(texts)

        enc = tok(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side="left",
        )
        response_tokens: torch.Tensor = enc["input_ids"]  # (B, T)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        attn = (response_tokens != pad_id).long()  # (B, T)
        B, T = response_tokens.shape

        if hasattr(secrets, "tolist"):  # numpy array -> list
            secrets = secrets.tolist()
        secrets = ["" if s is None else str(s) for s in secrets]
        sec = tok(secrets, add_special_tokens=False, padding=False, return_tensors=None)
        secret_ids_list = sec["input_ids"]  # List[List[int]]

        mask = torch.zeros_like(response_tokens)
        for i, needle_ids in enumerate(secret_ids_list):
            m = len(needle_ids)
            if m == 0:
                continue
            L = int(attn[i].sum().item())  # true (unpadded) length for this row
            if m > L:
                continue

            seg = response_tokens[i, -L:]  # right-aligned unpadded window
            wins = seg.unfold(0, m, 1)  # (L - m + 1, m)
            needle = torch.tensor(needle_ids, dtype=seg.dtype, device=seg.device)
            hits = (wins == needle).all(dim=1).nonzero(as_tuple=False).flatten()
            if hits.numel():
                pos = int(hits[0])  # or hits[-1] for "last"
                start = T - L + pos  # map back into left-padded coordinates
                mask[i, start : start + m] = 1
            else:
                raise RuntimeError(
                    f"[WARN] secret {secrets[i]!r} not found in sample {i}"
                )

        return response_tokens, mask

    def repeat(self, val: np.array, repeat_times: int = 2, interleave=False):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.
        """
        assert isinstance(val, np.ndarray), (
            f"val must be a numpy array, got {type(val)}"
        )
        if not interleave:
            val = np.repeat(val, repeat_times, axis=0)
        else:
            val = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return val

    def _delete_history_last_states(self, n: int) -> None:
        """
        Delete the last `n` states of the history.

        Args:
            n (int): Number of last states to delete.

        Raises:
            ValueError: If `n` is greater than the length of the history.
        """

        for hist in self.history:
            if len(hist) >= n:
                del hist[-n:]
            else:
                raise ValueError(
                    f"History length {len(hist)} is less than {n}, cannot delete last states."
                )

    def _replace_history_last_state(
        self, state: List[str], mask_np: List[bool], role: str
    ) -> None:
        """
        Replace the last state of the history with the user state.

        Args:
            state (List[str]): List of user states to replace the last state.
            mask_np (List[bool]): List of boolean masks indicating which history entries to update.
            role (str): Role of the user, e.g., "user" or "assistant".

        Raises:
            ValueError: If the length of `user_state` does not match the number of `True` values in `mask_np`.
        """

        idx = 0
        for mask, hist in zip(mask_np, self.history):
            if mask:
                hist[-1] = {"role": role, "content": state[idx]}
                idx += 1
            else:
                pass

    def _update_history(
        self,
        responses: List[str],
        mask_np: List[bool],
        role: str,
        parse_tag: str | None = None,
    ) -> None:
        """
        Update the history with the responses based on the mask.

        Args:
            responses (List[str]): List of responses to update the history with.
            mask_np (List[bool]): List of boolean masks indicating which history entries to update.
            role (str): Role of the user, e.g., "user" or "assistant".

        Raises:
            ValueError: If the length of `responses` does not match the number of `True` values in `mask_np`.
        """

        if parse_tag:
            parsed_responses = []
            for response in responses:
                content = _extract_question(response, parse_tag)
                parsed_responses.append(content)
            responses = parsed_responses

        idx = 0
        for mask, hist in zip(mask_np, self.history):
            if mask:
                hist.append({"role": role, "content": responses[idx]})
                idx += 1
            else:
                hist.append("")

    def retrieve_prompt(self, next_obs: List[str], current_step) -> List[str]:
        """
        Retrieve the prompt from the next observation.
        """
        obs_prompts = []
        for obs in next_obs:
            # Extract the question from the next observation
            obs_prompt = get_question_prompt(
                obs,
                current_step,
                thinking=self.config.actor_thinking,
                cot=self.config.actor_cot,
                max_questions=self.config.max_turns,
            )
            # Create a new prompt with the question
            obs_prompts.append(obs_prompt)
        return obs_prompts

    def run_game(
        self,
        raw_prompt: List[dict],
        gt: np.ndarray,
        n: int = 1,
        scenario: np.ndarray | str | None = None,
    ) -> DataProto:
        """
        Run main LLM generation loop.
        Args:
            raw_prompt: List[dict] - list of prompts
            gt: np.ndarray - ground truth secrets
            n: int - number of rollouts
            scenario: str - scenario name (for customer_service environment)

        Returns:
            DataProto - DataProto object containing the tokenized prompts

        Note:
         - active_mask is a boolean tensor of shape (bsz_all,), where True (1) indicates that the trajectory is active, False (0) indicates that the trajectory is finished
         - dones is a list of ints, where 1 indicates that the trajectory is finished, 0 indicates that the trajectory is active
        """
        bsz_single = len(raw_prompt)

        # Normalize scenario to a batch-length array (or None)
        if scenario is not None:
            scenario = np.array(scenario, dtype=object)
            if scenario.ndim == 0:
                scenario = np.full(bsz_single, scenario.item(), dtype=object)
            else:
                assert len(scenario) == bsz_single, (
                    f"scenario length must match batch size, got {len(scenario)} vs {bsz_single}"
                )

        # Rollouts
        if n > 1:
            raw_prompt = self.repeat(raw_prompt, repeat_times=n)
            gt = self.repeat(gt, repeat_times=n)
            scenario = (
                self.repeat(scenario, repeat_times=n) if scenario is not None else None
            )
            assert len(raw_prompt) == len(gt), (
                f"raw prompt and gt must have the same length, got {len(raw_prompt)} and {len(gt)}"
            )

        bsz_all = len(raw_prompt)
        group_ids = torch.arange(bsz_single).repeat_interleave(n) if n > 1 else None
        active_mask = torch.ones(
            bsz_all, dtype=torch.bool
        )  # create active mask for each sample
        turns_stats = torch.zeros(
            bsz_all, dtype=torch.int
        )  # set turn stats for each sample to 1
        valid_action_stats = torch.zeros(
            bsz_all, dtype=torch.int
        )  # set valid actions made for each sample to 0
        active_num_list = [active_mask.sum().item()]
        if self.config.logprob_reward.enabled or self.config.logprob_sampling.enabled:
            lp_secret = torch.zeros(
                (bsz_all, self.config.max_turns + 1), dtype=torch.float
            )  # bsz, max_turns+1
            lp_diff = torch.zeros(
                (bsz_all, self.config.max_turns), dtype=torch.float
            )  # bsz, max_turns
        else:
            lp_diff = None

        gt_rollings = gt
        judge_answer_shortest = None
        game_judge_metrics: Dict[str, int] = {key: 0 for key in JUDGE_METRICS_KEYS}
        game_judge_extras: Dict[str, int] = {key: 0 for key in JUDGE_EXTRAS_KEYS}

        raw_prompt = raw_prompt.tolist()
        self.history = [copy.copy(d) for d in raw_prompt]

        # system prompt + user prompt (without generation tokens)
        original_left_side = {"input_ids": self._create_left_side(self.history).long()}

        if self.config.logprob_reward.enabled or self.config.logprob_sampling.enabled:
            # base logprob for each secret
            logprob_scores = self.belief_log_prob(
                self.history,
                gt_rollings,
                return_shape=False,
                model=self.config.logprob_reward.base_model,
            )  # bsz and bsz,response_length
            assert logprob_scores.shape[0] == active_mask.sum(), (
                f"logprob scores should have the same batch size as active trj, got {logprob_scores.shape[0]} and {active_mask.sum()}"
            )
            lp_secret[:, 0] = logprob_scores

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():  # when all trajectories are finished
                break

            bsz_current = active_mask.sum()

            mask_np = active_mask.cpu().numpy()
            messages = [
                copy.deepcopy(h) for h, keep in zip(self.history, mask_np) if keep
            ]
            rollings = self._format_input(messages, tokenizer=self.tokenizer_actor)
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
                cut_left=True,
            )

            active_gt = gt_rollings[mask_np]
            active_scenario = (
                scenario
                if scenario is None
                else [s for s, keep in zip(scenario, mask_np) if keep]
            )
            assert active_mask.sum() == rollings.batch["input_ids"].shape[0], (
                "active mask and rolling state must match"
            )

            turns_stats[active_mask] += 1

            if self.config.debug:
                print(f"[DEBUG] Game round {step + 1} \n")
                input_str = self.tokenizer_actor.decode(
                    rollings.batch["input_ids"][0], skip_special_tokens=False
                )
                print("[DEBUG] Current input prompt to actor \n", input_str)

            assert rollings.batch["input_ids"].shape[0] == len(active_gt), (
                "Active trajectory len must match the number of secrets passed"
            )

            # Actor
            # NOTE: in the vllm generation we do not use the attention mask,
            #       rather we always MUST use left padding, so the input passed is correct
            # extract the rollout fanout from the meta_info
            rollout_fanout = self.actor_rollout_wg.get_rollout_fanout(
                rollings.meta_info["validate"]
            )[0]
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                rollings, self.actor_rollout_wg.world_size
            )  # bsz, prompt_len
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            meta_info = test_output_gen_batch_padded.meta_info
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded,
                pad_size=pad_size,
                rollout_fanout=rollout_fanout,
            )
            responses_ids, responses_str = self._postprocess_responses(
                test_output_gen_batch.batch["responses"]
            )

            if self.config.debug:
                print(f"[DEBUG] Actor responses at step {step}:\n")
                max_dbg = min(3, len(responses_str))
                for i in range(max_dbg):
                    print(f"--- Response {i} ---\n", responses_str[i])
                    print("------------------\n")

            # Pick the sampled response with the highest logprob diff for every batch item
            if self.config.logprob_sampling.enabled and rollout_fanout > 1:
                if self.config.debug:
                    print(
                        f"[DEBUG] Performing logprob-based sampling with rollout_fanout={rollout_fanout} for step {step}"
                    )
                n_logprob_samples = rollout_fanout

                responses_ids = responses_ids.reshape(
                    bsz_current, n_logprob_samples, -1
                )
                responses_str = np.array(
                    [
                        responses_str[i : i + n_logprob_samples]
                        for i in range(0, len(responses_str), n_logprob_samples)
                    ]
                )

                # This round's log probability differences for each sample
                round_lp_secret = torch.zeros(
                    (bsz_all, n_logprob_samples), dtype=torch.float
                )

                # Judge's cached responses
                cached_next_obs = []
                cached_dones = []
                cached_valid_action = []
                cached_answer_str = []
                cached_judge_metrics: Dict[str, List[List[int]]] = {
                    key: [] for key in JUDGE_METRICS_KEYS
                }
                cached_judge_extras: Dict[str, List[List[int]]] = {
                    key: [] for key in JUDGE_EXTRAS_KEYS
                }

                # Calculate log probability differences for each sample, for all batch items
                for i in range(n_logprob_samples):
                    sample_ids = responses_ids[:, i, :]  # bsz, response_length
                    sample_str = responses_str[:, i].tolist()  # bsz, response_length

                    # Actor simulation
                    self._update_history(
                        sample_str,
                        mask_np,
                        "assistant",
                        self.config.actor_thinking and "question" or None,
                    )

                    # make sure that the message does not contain the last assistant update
                    first_active_idx = int(np.flatnonzero(mask_np)[0])
                    assert (
                        len(messages[0]) == len(self.history[first_active_idx]) - 1
                    ), (
                        f"Length mismatch: msg len={len(messages[0])}, "
                        f"history len={len(self.history[first_active_idx])}"
                    )

                    # Oracle simulation
                    (
                        next_obs,
                        dones,
                        valid_action,
                        answer_str,
                        turn_judge_metrics,
                        turn_judge_extras,
                    ) = self._ask_question(
                        sample_ids,
                        sample_str,
                        active_gt,
                        history=messages if self.config.repeated_prompt else None,
                        thinking=self.config.actor_thinking,
                        cot=self.config.actor_cot,
                    )
                    cached_next_obs.append(next_obs)
                    cached_dones.append(dones)
                    cached_valid_action.append(valid_action)
                    cached_answer_str.append(answer_str)
                    for key in turn_judge_metrics.keys():
                        if key not in cached_judge_metrics:
                            cached_judge_metrics[key] = []
                        cached_judge_metrics[key].append(turn_judge_metrics[key])
                    for key in turn_judge_extras.keys():
                        if key not in cached_judge_extras:
                            cached_judge_extras[key] = []
                        cached_judge_extras[key].append(turn_judge_extras[key])

                    self._update_history(next_obs, mask_np, "user")

                    # Calculate log probability differences
                    messages_logprob = [
                        copy.deepcopy(h)
                        for h, keep in zip(self.history, mask_np)
                        if keep
                    ]
                    lp_scores = self.belief_log_prob(
                        messages_logprob, active_gt, return_shape=False, model="actor"
                    )  # bsz and bsz,response_length
                    assert lp_scores.shape[0] == active_mask.sum(), (
                        f"logprob scores should have the same batch size as active trj, got {lp_scores.shape[0]} and {active_mask.sum()}"
                    )
                    round_lp_secret[mask_np, i] = lp_scores

                    # Restore the history
                    self._delete_history_last_states(2)

                first_active_idx = np.where(mask_np)[0][0].item()

                # Diversity-based logprob sampling
                best_n = min(self.config.logprob_sampling.best_n, n)
                worst_n = min(self.config.logprob_sampling.worst_n, n - best_n)
                assert best_n + worst_n <= n, (
                    f"best_n + worst_n must be less than or equal \
                    to n, got {best_n} + {worst_n} > {n}"
                )
                round_diff = torch.zeros(
                    (bsz_all, n_logprob_samples), dtype=torch.float
                )
                round_diff[mask_np] = round_lp_secret[mask_np] - lp_secret[
                    mask_np, step
                ].unsqueeze(1)  # step wise

                chosen_sample_idxs = torch.zeros(bsz_all, dtype=torch.long)

                if n > 1 and (best_n + worst_n) > 0:
                    rollout_indices = torch.arange(bsz_all) % n
                    best_mask = rollout_indices < best_n
                    worst_mask = (rollout_indices >= best_n) & (
                        rollout_indices < best_n + worst_n
                    )
                    random_mask = rollout_indices >= best_n + worst_n

                    if best_mask.any():
                        chosen_sample_idxs[best_mask] = round_diff[best_mask, :].argmax(
                            dim=1
                        )
                    if worst_mask.any():
                        chosen_sample_idxs[worst_mask] = round_diff[
                            worst_mask, :
                        ].argmin(dim=1)
                    if random_mask.any():
                        chosen_sample_idxs[random_mask] = torch.randint(
                            0, n_logprob_samples, (random_mask.sum().item(),)
                        )

                # in case of no groups also allow mixed sampling strategy
                else:
                    coin = torch.rand(bsz_all, device=round_diff.device)
                    pick_best_mask = coin < self.config.logprob_sampling.p_best
                    pick_rand_mask = ~pick_best_mask

                    if pick_best_mask.any():
                        chosen_sample_idxs[pick_best_mask] = round_diff[
                            pick_best_mask, :
                        ].argmax(dim=1)

                    if pick_rand_mask.any():
                        chosen_sample_idxs[pick_rand_mask] = torch.randint(
                            0,
                            n_logprob_samples,
                            (int(pick_rand_mask.sum().item()),),
                            device=round_diff.device,
                        )

                chosen_sample_idxs = chosen_sample_idxs[active_mask]
                responses_ids = responses_ids[
                    torch.arange(bsz_current), chosen_sample_idxs
                ]
                responses_str = responses_str[
                    np.arange(bsz_current), chosen_sample_idxs.detach().cpu().numpy()
                ].tolist()

                # Actor update
                self._update_history(
                    responses_str,
                    mask_np,
                    "assistant",
                    self.config.actor_thinking and "question" or None,
                )

                # Select the oracle results based on chosen_sample_idxs
                next_obs = [
                    cached_next_obs[idx][i] for i, idx in enumerate(chosen_sample_idxs)
                ]
                dones = [
                    cached_dones[idx][i] for i, idx in enumerate(chosen_sample_idxs)
                ]
                valid_action = [
                    cached_valid_action[idx][i]
                    for i, idx in enumerate(chosen_sample_idxs)
                ]
                for key, value in cached_judge_metrics.items():
                    if key not in game_judge_metrics:
                        game_judge_metrics[key] = 0
                    game_judge_metrics[key] += sum(
                        value[idx][i] for i, idx in enumerate(chosen_sample_idxs)
                    )
                for key, value in cached_judge_extras.items():
                    if key not in game_judge_extras:
                        game_judge_extras[key] = 0
                    game_judge_extras[key] += sum(
                        value[idx][i] for i, idx in enumerate(chosen_sample_idxs)
                    )

                answer_str = cached_answer_str[0]

                # Oracle update
                self._update_history(next_obs, mask_np, "user")

                sel = chosen_sample_idxs.unsqueeze(1)  # (B_active, 1)
                lp_secret[mask_np, step + 1] = (
                    round_lp_secret[mask_np].gather(1, sel).squeeze(1)
                )  # bsz, = sum(mask_np)

                # Only handle active trajectories & filter out finished
                turn_diff = round_diff[mask_np].gather(1, sel).squeeze(1).clone()

                if (
                    self.config.logprob_reward.normalised
                    and "episode_centering" not in self.config.logprob_reward.methods
                ):
                    # check which trajectories are still not 'Finished' (1=finihsed, 0-active), therefore flip for nan creation
                    finished = torch.as_tensor(
                        dones, dtype=torch.bool, device=turn_diff.device
                    )  # bsz = sum(mask_np)
                    turn_diff = torch.where(
                        finished, torch.full_like(turn_diff, torch.nan), turn_diff
                    )  # bsz = sum(mask_np)

                    if "batch_centering" in self.config.logprob_reward.methods:
                        if "batch_std" in self.config.logprob_reward.methods:
                            std_mode = "batch"
                        else:
                            std_mode = None
                        turn_diff = self._normalize_logprobs(
                            turn_diff,
                            mean_mode="batch",
                            std_mode=std_mode,
                            group_ids=None,
                        )  # bsz, = sum(mask_np)

                    elif (
                        "group_centering" in self.config.logprob_reward.methods
                        and group_ids is not None
                    ):
                        if "group_std" in self.config.logprob_reward.methods:
                            std_mode = "group"
                        elif "batch_std" in self.config.logprob_reward.methods:
                            std_mode = "batch"
                        else:
                            std_mode = None
                        turn_diff = self._normalize_logprobs(
                            turn_diff,
                            mean_mode="group",
                            std_mode=std_mode,
                            group_ids=group_ids[mask_np],
                        )  # bsz, = sum(mask_np)

                    elif (
                        "group_min_max" in self.config.logprob_reward.methods
                        and group_ids is not None
                    ):
                        # pacr approach: pos only and group min max normalisation
                        turn_diff = torch.relu(turn_diff)

                        turn_diff = self._normalize_logprobs(
                            turn_diff,
                            mean_mode="group_min_max",
                            std_mode=None,
                            group_ids=group_ids[mask_np],
                        )  # bsz, = sum(mask_np)

                    # Check if any of the clipping methods are enabled, they are exclusive
                    if "tanh" in self.config.logprob_reward.methods:
                        # simple tanh clamping, no normalisation
                        turn_diff = self._squash_logprobs(turn_diff, "tanh")

                    elif "sigmoid" in self.config.logprob_reward.methods:
                        turn_diff = self._squash_logprobs(turn_diff, "sigmoid")

                    elif "min_max" in self.config.logprob_reward.methods:
                        turn_diff = turn_diff.clamp(
                            self.config.logprob_reward.clipping["min"],
                            self.config.logprob_reward.clipping["max"],
                        )

                    elif "positive" in self.config.logprob_reward.methods:
                        turn_diff = torch.relu(turn_diff)

                    # assign zero to nan diffs (finished episodes) after normalisation
                    turn_diff = torch.where(
                        torch.isnan(turn_diff), torch.zeros_like(turn_diff), turn_diff
                    )

                # keep the attention mask of each turn to turn the correct response later
                lp_diff[mask_np, step] = turn_diff.detach()

            # Do not make a choice based on log probability sampling, just use the first response for each batch item
            else:
                if (
                    responses_ids.shape[0] != bsz_current
                    or len(responses_str) != bsz_current
                ):
                    raise ValueError(
                        f"Expected {bsz_current} responses, got {responses_ids.shape[0]} (ids) and {len(responses_str)} (str)"
                    )

                # Actor update
                self._update_history(
                    responses_str,
                    mask_np,
                    "assistant",
                    self.config.actor_thinking and "question" or None,
                )

                # make sure that the message does not contain the last assistant update
                # Find the corresponding active trajectory in self.history for comparison
                first_active_idx = int(np.flatnonzero(mask_np)[0])  # first active index
                assert len(messages[0]) == len(self.history[first_active_idx]) - 1, (
                    f"Length mismatch: msg len={len(messages[0])}, "
                    f"history len={len(self.history[first_active_idx])}"
                )

                # Oracle update
                # Pass only the subset data to the judge, get out judge predictions in List[dict] formatted as chat
                (
                    next_obs,
                    dones,
                    valid_action,
                    answer_str,
                    turn_judge_metrics,
                    turn_judge_extras,
                ) = self._ask_question(
                    responses_ids,
                    responses_str,
                    active_gt,
                    history=messages if self.config.repeated_prompt else None,
                    thinking=self.config.actor_thinking,
                    cot=self.config.actor_cot,
                    scenarios=active_scenario,
                )

                # Update judge metrics & extras
                for key, value in turn_judge_metrics.items():
                    if key not in game_judge_metrics:
                        game_judge_metrics[key] = 0
                    game_judge_metrics[key] += sum(value)
                for key, value in turn_judge_extras.items():
                    if key not in game_judge_extras:
                        game_judge_extras[key] = 0
                    game_judge_extras[key] += sum(value)

                self._update_history(next_obs, mask_np, "user")

                # Only calculate the logprob diffs if we want to use logprob rewards
                # BUT NOT logprob sampling
                if self.config.logprob_reward.enabled:
                    messages = [
                        copy.deepcopy(h)
                        for h, keep in zip(self.history, mask_np)
                        if keep
                    ]
                    logprob_scores = self.belief_log_prob(
                        messages, active_gt, model=self.config.logprob_reward.step_model
                    )  # bsz and bsz,response_length

                    assert logprob_scores.shape[0] == active_mask.sum(), (
                        f"logprob scores should have the same batch size as active trj, got {logprob_scores.shape[0]} and {active_mask.sum()}"
                    )

                    # If base and step model are the same, we can optimize by reusing previous computations
                    if (
                        self.config.logprob_reward.step_model
                        == self.config.logprob_reward.base_model
                    ):
                        lp_secret[mask_np, step + 1] = logprob_scores
                        # Only handle active trajectories & filter out finished
                        turn_diff = (
                            lp_secret[mask_np, step + 1] - lp_secret[mask_np, step]
                        )  # shape = sum(mask_np)
                    else:
                        assert self.config.logprob_reward.base_model == "ref", (
                            "When using different step and base models, the base model must be 'ref'"
                        )

                        # compute the ema difference actor - ema
                        turn_diff = logprob_scores - lp_secret[mask_np, step]

                        # compute the next step for the ema model (to store for the next step)
                        lp_secret[mask_np, step + 1] = self.belief_log_prob(
                            messages,
                            active_gt,
                            model=self.config.logprob_reward.base_model,
                        )  # bsz and bsz,response_length

                    if (
                        self.config.logprob_reward.normalised
                        and "episode_centering"
                        not in self.config.logprob_reward.methods
                    ):
                        # check which trajectories are still not 'Finished' (1=finihsed, 0-active), therefore flip for nan creation
                        finished = torch.as_tensor(
                            dones, dtype=torch.bool, device=turn_diff.device
                        )  # bsz = sum(mask_np)
                        turn_diff = torch.where(
                            finished, torch.full_like(turn_diff, torch.nan), turn_diff
                        )  # bsz = sum(mask_np)

                        if "batch_centering" in self.config.logprob_reward.methods:
                            if "batch_std" in self.config.logprob_reward.methods:
                                std_mode = "batch"
                            else:
                                std_mode = None

                            # Normalize and squash logprob differences
                            turn_diff = self._normalize_logprobs(
                                turn_diff,
                                mean_mode="batch",
                                std_mode=std_mode,
                                group_ids=None,
                            )  # bsz, = sum(mask_np)

                        elif (
                            "group_centering" in self.config.logprob_reward.methods
                            and group_ids is not None
                        ):
                            if "group_std" in self.config.logprob_reward.methods:
                                std_mode = "group"
                            elif "batch_std" in self.config.logprob_reward.methods:
                                std_mode = "batch"
                            else:
                                std_mode = None
                            turn_diff = self._normalize_logprobs(
                                turn_diff,
                                mean_mode="group",
                                std_mode=std_mode,
                                group_ids=group_ids[mask_np],
                            )  # bsz, = sum(mask_np)

                        elif (
                            "group_min_max" in self.config.logprob_reward.methods
                            and group_ids is not None
                        ):
                            # pacr approach: pos only and group min max normalisation
                            turn_diff = torch.relu(turn_diff)
                            turn_diff = self._normalize_logprobs(
                                turn_diff,
                                mean_mode="group_min_max",
                                std_mode=None,
                                group_ids=group_ids[mask_np],
                            )  # bsz, = sum(mask_np)

                        # Check if any of the clipping methods are enabled, they are exclusive
                        if "tanh" in self.config.logprob_reward.methods:
                            # simple tanh clamping, no normalisation
                            turn_diff = self._squash_logprobs(turn_diff, "tanh")
                        elif "sigmoid" in self.config.logprob_reward.methods:
                            turn_diff = self._squash_logprobs(turn_diff, "sigmoid")
                        elif "min_max" in self.config.logprob_reward.methods:
                            turn_diff = turn_diff.clamp(
                                self.config.logprob_reward.clipping["min"],
                                self.config.logprob_reward.clipping["max"],
                            )
                        elif "positive" in self.config.logprob_reward.methods:
                            turn_diff = torch.relu(turn_diff)

                    # assign zero to nan diffs (finished episodes) after normalisation
                    turn_diff = torch.where(
                        torch.isnan(turn_diff), torch.zeros_like(turn_diff), turn_diff
                    )
                    lp_diff[mask_np, step] = turn_diff.detach()

                    if self.config.debug:
                        print(f"[DEBUG] Game turn {step}, logprob computation \n")
                        print(f"[DEBUG] len messages: {len(messages)}")
                        print(f"[DEBUG] active_gt: {len(active_gt)}")
                        print(f"[DEBUG] logprob scores shape: {logprob_scores.shape}")
                        print(
                            f"[DEBUG] Logprob for first sample: {lp_secret[mask_np, step][0]} \n"
                        )
                        print(f"[DEBUG] norm lp_diff: {lp_diff[0]}")  # tokens
                        print(
                            f"[DEBUG] mean of norm_logprobs_diff turn: {turn_diff.mean()}"
                        )
                        print(
                            f"[DEBUG] std of norm_logprobs_diff turn: {turn_diff.std()}"
                        )

            assert responses_ids.shape[0] == bsz_current, (
                f"responses_ids must have the same batch size as raw prompt, got {responses_ids.shape[0]} and {bsz_current}"
            )
            assert len(responses_str) == bsz_current, (
                f"responses_str must have the same lenght as raw prompt, got {len(responses_str)} and {bsz_current}"
            )

            # Update active mask
            curr_active_mask = self.tensor_fn._pad_active(dones, active_mask)
            valid_action = self.tensor_fn._pad_action(valid_action, active_mask)
            active_mask = active_mask * curr_active_mask  # update active mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += valid_action

            if (
                active_num_list[-1] < active_num_list[-2]
                and judge_answer_shortest is None
            ):
                judge_answer_shortest = answer_str

        torch.cuda.empty_cache()

        meta_info["turns_stats"] = turns_stats.tolist()
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["active_mask"] = [int(m) for m in meta_info["active_mask"]]

        meta_info["judge_metrics"] = game_judge_metrics
        meta_info["judge_extras"] = game_judge_extras

        if self.config.logprob_reward.enabled:
            # apply normalistion to the lp_diff if enabled
            if (
                self.config.logprob_reward.normalised
                and "episode_centering" in self.config.logprob_reward.methods
            ):
                from delta_belief_rl.utils.format import episode_centered_diff

                lp_diff = episode_centered_diff(lp_diff)

                if "tanh" in self.config.logprob_reward.methods:
                    # simple tanh clamping, no normalisation
                    lp_diff = self._squash_logprobs(lp_diff, "tanh")
                    print("lp_diff after tanh:", lp_diff[0])
                elif "sigmoid" in self.config.logprob_reward.methods:
                    lp_diff = self._squash_logprobs(lp_diff, "sigmoid")
                    print("lp_diff after sigmoid:", lp_diff[0])
                elif "min_max" in self.config.logprob_reward.methods:
                    lp_diff = torch.clamp(
                        lp_diff,
                        self.config.logprob_reward.clipping["min"],
                        self.config.logprob_reward.clipping["max"],
                    )
                    print("lp_diff after clipping:", lp_diff[0])

            meta_info["logprob"] = lp_secret.tolist()
            meta_info["logprob_diff"] = lp_diff.tolist()

        if self.config.debug:
            print("ACTIVE_TRAJ_NUM:", active_num_list)
            success_rate = 1.0 - (
                np.sum(meta_info["active_mask"]) / len(meta_info["active_mask"])
            )
            print(f"[DEBUG] Success rate: {success_rate:.2f}%")

        if judge_answer_shortest is None:
            # none of the trajectories finished
            judge_answer_shortest = answer_str

        # update the final response
        right_side = self._format_responses_loss(original_left_side)

        return self._compose_final_output(
            original_left_side,
            right_side,
            meta_info,
            gt_rollings,
            judge_answer_shortest,
            lp_diff,
        )

    def belief_log_prob(
        self,
        messages: List[dict],
        gt: np.ndarray[str],
        return_shape: bool = False,
        model: str = "actor",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the log probabilities on the fly.

        Args:
            messages: List of conversation histories (each is a List[{"role": str, "content": str}, ...]).
            gt: Ground-truth secrets aligned 1:1 with `messages`.
            return_shape: Whether to return the formatted "original" last-two-state inputs.
            model: Which model to use for logprob computation ('actor' or 'ref').

        Returns:
            logprob_scores: Tensor of log-probabilities for the secret word (per batch item).
            rollings_orig: Original last-two-state rollings (or None if return_shape=False).
        """
        bsz = len(messages)
        assert bsz == len(gt), (
            f"messages and gt must have the same length, got {bsz} and {len(gt)}"
        )

        # Precompute elicitation prompts once (saves repeated function calls in the loop)
        elicitation_prompts = [get_elicitation(secret) for secret in gt]

        # Build a fresh augmented history list instead of copying/mutating dicts
        # (avoids O(n) shallow copies and potential side effects)
        messages_with_elicitation = [
            hist + [{"role": "assistant", "content": elicitation_prompts[i]}]
            for i, hist in enumerate(messages)
        ]

        # Encode the augmented histories (we are not generating; just encoding)
        rollings = self._format_input(
            messages_with_elicitation,
            generation_prompt=False,
            tokenizer=self.tokenizer_actor,
        )

        # Create responses + mask for selecting only secret-word tokens
        elicitation_responses, elicitation_mask = self._create_responses_mask(
            messages_with_elicitation, gt, tok=self.tokenizer_actor
        )

        # Merge responses into rollings (one union; avoids multiple small unions/appends later)
        rollings = rollings.union(
            DataProto.from_dict(
                {
                    "responses": elicitation_responses,  # we pass the responses to filrer out only these tokens from the logprobs
                    "elicitation_mask": elicitation_mask,
                }
            )
        )

        if self.config.debug:
            L = elicitation_mask.shape[1]
            try:
                print(f"[DEBUG] messages tail: {messages_with_elicitation[0][-3:]}")
                print(
                    f"[DEBUG] rollings ids tail: {rollings.batch['input_ids'][0, -L:]}"
                )
                print(f"[DEBUG] elicitation_mask[0]: {elicitation_mask[0]}")
                text = self.tokenizer_actor.decode(
                    rollings.batch["input_ids"][0, -L:][elicitation_mask[0] == 1],
                    skip_special_tokens=False,
                )
                print(f"[DEBUG] tokens under mask: {text}")
            except Exception as e:
                print(f"[DEBUG] debug section error: {e}")

        # Compute log-probs with padding to world size
        rollings_padded, pad_size = pad_dataproto_to_divisor(
            rollings, self.actor_rollout_wg.world_size
        )
        # Set meta for logprob computation (avoid mutating shared meta by copying only what's needed)
        rollings_padded.meta_info["logprob_secret"] = {
            "temperature": 1,
            "calculate_entropy": False,
        }

        if model == "ref":
            logprobs_padded = self.actor_rollout_wg.compute_ref_log_prob(
                rollings_padded
            )  # bsz x response_len
        else:
            logprobs_padded = self.actor_rollout_wg.compute_log_prob(
                rollings_padded
            )  # bsz x response_len
        logprobs = unpad_dataproto(logprobs_padded, pad_size=pad_size)

        # Aggregate over multiple tokens using the mask
        logprob_scores = self.get_scores(
            logprobs.batch["log_probs"],
            elicitation_mask,
            gt,
            elicitation_responses,
            tokenizer=self.tokenizer_actor,
        )

        return logprob_scores

    def get_scores(
        self,
        log_probs: torch.Tensor,
        response_attention_mask: torch.Tensor,
        gts: List[str],
        response_token_ids: torch.Tensor,
        ctx_window: int = 2,
        tokenizer=None,
    ) -> torch.Tensor:
        """
        array of log probs for potentially multiple tokens composing the secret
        --> single reward value per trajectory

        Note: Overwrites the input log_probs tensor.
        """

        B, L = log_probs.shape
        assert response_attention_mask.shape == (B, L)
        assert response_token_ids.shape == (B, L)

        mask = response_attention_mask == 1

        if self.config.debug:
            b = 0
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                print(
                    f"[batch {b}] (no masked tokens) gt={gts[b] if b < len(gts) else 'N/A'}"
                )

            # Print the exact masked positions first
            toks_ids = response_token_ids[b, idx]
            vals = log_probs[b, idx]
            toks_str = [
                tokenizer.decode([tid.item()], skip_special_tokens=False)
                for tid in toks_ids
            ]
            print(
                f"[batch {b}] masked idx={idx.tolist()} toks={toks_str} lp={vals.detach().cpu().tolist()} gt={gts[b] if b < len(gts) else 'N/A'}"
            )

            # Now inspect adjacent context per contiguous span
            spans = self.tensor_fn._contiguous_spans(idx)
            for s in spans:
                left = max(0, s[0].item() - ctx_window)
                right = min(L, s[-1].item() + ctx_window + 1)  # exclusive
                pos = torch.arange(left, right, device=log_probs.device)

                ctx_ids = response_token_ids[b, pos]
                ctx_vals = log_probs[b, pos]
                ctx_mask = mask[b, pos]
                ctx_tokens = [
                    tokenizer.decode([tid.item()], skip_special_tokens=False)
                    for tid in ctx_ids
                ]

                # Pretty one-line dump per context window
                row = []
                for j, p in enumerate(pos.tolist()):
                    mark = "*" if ctx_mask[j].item() else " "  # star = inside mask
                    row.append(f"{mark}{p}:{ctx_tokens[j]!r}:{float(ctx_vals[j]):.4f}")
                print(
                    f"[batch {b}] span {s.tolist()}  ctx[{left}:{right}] -> "
                    + " | ".join(row)
                )

        selected = torch.where(mask, log_probs, torch.zeros_like(log_probs))

        if self.config.logprob_reward.agg_method == "sum":
            out = selected.sum(dim=1)
        elif self.config.logprob_reward.agg_method == "mean":
            counts = mask.sum(dim=1)
            out = selected.sum(dim=1) / counts.clamp_min(1)
            no_sel = counts == 0
            if no_sel.any():
                out = torch.where(no_sel, torch.full_like(out, float("-inf")), out)
        else:
            raise ValueError(
                f"Unknown logprob aggregation method: {self.config.logprob_agg}"
            )

        return out

    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        meta_info: Dict,
        gt: List[str],
        judge_answer: str,
        lp_diff: torch.Tensor | None = None,
    ) -> DataProto:
        """Compose final generation output."""

        final_output = right_side.copy()  # copy the right side to the final output
        final_output["prompts"] = left_side[
            "input_ids"
        ]  # this is with the assistant generation token (and no think)
        final_output["input_ids"] = torch.cat(
            [left_side["input_ids"], right_side["responses"]], dim=1
        )

        # Create attention mask and position ids
        final_output["attention_mask"] = self.tensor_fn.create_attention_mask(
            final_output["input_ids"]
        )

        final_output["position_ids"] = self.tensor_fn.create_position_ids(
            final_output["attention_mask"]
        )

        if lp_diff is not None:
            final_output["elicit_reward"] = lp_diff  # bsz, max_turns

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        final_output.non_tensor_batch["history"] = self.history

        if self.config.debug:
            print(
                f"[DEBUG] final output {final_output.batch}, {final_output.non_tensor_batch.keys()}"
            )
            idx = [
                i for i, m in enumerate(final_output.meta_info["active_mask"]) if not m
            ]
            if not idx:
                idx = 0
            elif len(idx) == 1:
                idx = idx[0]
            else:
                idx = min(idx, key=lambda i: final_output.meta_info["turns_stats"][i])
            print(f"secret: {gt[idx]}")
            input_str = self.tokenizer_actor.decode(
                final_output.batch["responses"][idx], skip_special_tokens=True
            )
            print(f" Final output response + info: {input_str}")

        return final_output

    def _normalize_logprobs(
        self,
        logprobs: torch.Tensor,
        mean_mode: str | None,
        std_mode: str | None,
        group_ids: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize logprobs according to specified mean/std modes.

        Args:
            logprobs: (bsz,) of only active trajectories (sum(mask_np)) of logprob_diff
            mean_mode: 'batch', 'group', or None
            std_mode: 'batch', 'group', or None
            group_ids: (bsz,) tensor of group IDs (required if group mode is used)
            eps: small constant to avoid division by zero

        Returns:
            (bsz,) tensor of normalized logprobs of only active trajectories (sum(mask_np))
        """
        out = logprobs.clone()

        # Early return if everything is NaN
        if torch.isnan(out).all():
            return out

        # only apply gorup mode id group_ids is not None and mean_mode is "group"
        if group_ids is not None and mean_mode == "group":
            group_ids = torch.as_tensor(group_ids, device=out.device)
            for g in group_ids.unique():
                mask = group_ids == g
                gv = out[mask]
                if torch.isnan(gv).all():
                    continue
                gm = torch.nanmean(gv)
                out[mask] = gv - gm
        elif mean_mode == "batch":
            mean = out.nanmean()
            out = out - mean

        elif group_ids is not None and mean_mode == "group_min_max":
            group_ids = torch.as_tensor(group_ids, device=out.device)
            for g in group_ids.unique():
                mask = group_ids == g
                gv = out[mask]
                if torch.isnan(gv).all():
                    continue
                valid = gv[~torch.isnan(gv)]
                # PyTorch < 2.1 does not expose torch.nanmin/torch.nanmax.
                # Fall back to masking out NaNs before taking extrema.
                if valid.numel() == 0:
                    continue
                gmin = valid.min()
                gmax = valid.max()
                out[mask] = (gv - gmin) / (gmax - gmin + eps)

        if std_mode == "batch":
            count = (~torch.isnan(out)).sum().item()
            if count > 1:
                std = nanstd(out, unbiased=False)
                if torch.isfinite(std):
                    out = out / std.clamp_min(eps)

        elif group_ids is not None and std_mode == "group":
            group_ids = torch.as_tensor(group_ids, device=out.device)
            for g in group_ids.unique():
                mask = group_ids == g
                gv = out[mask]
                count = (~torch.isnan(gv)).sum().item()
                if count > 1:
                    std = nanstd(gv, unbiased=False)
                    if torch.isfinite(std):
                        out[mask] = gv / std.clamp_min(eps)

        return out

    def _squash_logprobs(
        self, logprobs: torch.Tensor, squash_method: str, fill_nan: float | None = 0.0
    ) -> torch.Tensor:
        """
        Squash logprobs according to specified type.

        Args:
            logprobs: (bs,) tensor of logprobs
            squash_method: 'tanh', 'sigmoid', or None

        Returns:
            (bs,) tensor of squashed logprobs
        """
        if fill_nan is not None:
            logprobs = logprobs.nan_to_num(nan=fill_nan)

        if squash_method == "tanh":
            return torch.tanh(logprobs * self.config.logprob_reward.tau)
        elif squash_method == "sigmoid":
            return torch.sigmoid(logprobs)
        else:
            raise ValueError(f"Unknown squash type: {squash_method}")
