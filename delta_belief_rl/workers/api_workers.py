# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
API-based Judge Rollout Worker using external API calls instead of local inference.

Example configuration:
    judge:
        model:
            api_model_name: "qwen/qwen3-235b-a22b-2507"  # OpenRouter model name
            temperature: 0.7
            top_p: 0.9
        api_batch_size: 200  # Number of prompts to process per batch
        thinking: false  # Enable thinking tags in prompts
        cot: cot  # Chain-of-thought reasoning (mutually exclusive with thinking)
"""

import logging
import os
import re
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from delta_belief_rl.llm_agent.prompts import get_judge_prompt, get_judge_system_prompt
from delta_belief_rl.workers.rollout.api_batch_processing import batch_send
import asyncio

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# Find all <answer>…</answer> spans, case‐insensitive, across lines
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# Remove any characters except letters, digits, dot, comma, and space
CLEAN_PATTERN = re.compile(r"[^a-zA-Z0-9\., ]+")


class APIJudgeRolloutWorker(Worker):
    """
    This worker uses API calls instead of local model inference for judge rollout.
    It provides the same functionality as JudgeRolloutWorker but relies on external APIs.
    """

    def __init__(self, config: DictConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # Load tokenizer - either passed directly or from config path
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif hasattr(config.model, "tokenizer_path") and config.model.tokenizer_path:
            # Load tokenizer from path specified in config
            from verl.utils.fs import copy_to_local

            local_path = copy_to_local(config.model.tokenizer_path)
            trust_remote_code = config.model.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(
                local_path, trust_remote_code=trust_remote_code
            )
            logger.info(f"Loaded tokenizer from {config.model.tokenizer_path}")
        else:
            self.tokenizer = None
            logger.warning(
                "No tokenizer provided - API worker can only handle text input, not token IDs"
            )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        # Extract model configuration for API calls
        self.model_config = {
            "name": self.config.model.get(
                "api_model_name", "qwen/qwen3-30b-a3b-instruct-2507"
            ),
            "temperature": self.config.model.get("temperature", 0.7),
            "top_p": self.config.model.get("top_p", 0.9),
        }

        # API batch size for processing
        self.api_batch_size = self.config.get("api_batch_size", 200)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize API worker components."""
        logger.info(
            f"Initializing API Judge Worker with model: {self.model_config['name']}"
        )
        logger.info(f"API batch size: {self.api_batch_size}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_rollout(self):
        """Prepare for rollout."""
        logger.info("Starting API judge rollout (Text Input)")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def exit_rollout(self):
        """Clean up after rollout."""
        logger.info("Exiting API judge rollout (Text Input)")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        Generate sequences using API calls.

        Can accept either:
        - Token IDs in prompts.batch['input_ids'] (decoded using tokenizer)
        - Text in prompts.non_tensor_batch['questions']

        Args:
            prompts: DataProto containing either token IDs or question texts and metadata

        Returns:
            DataProto with generated responses and extracted answers
        """
        # Extract questions - either from text or by decoding token IDs
        questions = prompts.non_tensor_batch.get("questions")

        if questions is None:
            # Decode from input_ids
            if "input_ids" not in prompts.batch:
                raise ValueError(
                    "APIJudgeRolloutWorker requires either 'questions' in non_tensor_batch "
                    "or 'input_ids' in batch"
                )
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is required when input_ids are provided. "
                    "Pass tokenizer to APIJudgeRolloutWorker.__init__()"
                )
            input_ids = prompts.batch["input_ids"]
            questions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Extract ground truth and scenario/env context
        gt_str = prompts.non_tensor_batch.get("golden_answers")
        if gt_str is None:
            gt_str = prompts.non_tensor_batch.get("ground_truth")

        if gt_str is None:
            raise ValueError(
                "APIJudgeRolloutWorker requires 'golden_answers' or 'ground_truth' "
                "in prompts.non_tensor_batch"
            )

        scenarios = prompts.non_tensor_batch.get("scenarios", None)
        env = self.config.get("env", "twenty_questions")

        # Determine batch size
        batch_size = len(questions)

        history = prompts.meta_info.get("history", None)
        prompt_texts = []
        for i in range(batch_size):
            if history is not None and len(history) > i:
                prompt_content = get_judge_prompt(
                    gt_str[i],
                    questions[i],
                    history[i],
                    thinking=self.config.thinking,
                    cot=self.config.cot,
                    env=env,
                    scenario=scenarios[i] if scenarios is not None else None,
                )
            else:
                prompt_content = get_judge_prompt(
                    gt_str[i],
                    questions[i],
                    None,
                    thinking=self.config.thinking,
                    cot=self.config.cot,
                    env=env,
                    scenario=scenarios[i] if scenarios is not None else None,
                )

            prompt_texts.append(prompt_content)

        # Get system prompt
        system_prompt = get_judge_system_prompt(
            repeated=(history is not None),
            env=env,
        )

        # Make API calls in batches
        logger.info(f"Processing {len(prompt_texts)} prompts via API (Text Input)")
        api_responses = []
        for i in range(0, len(prompt_texts), self.api_batch_size):
            batch_prompts = prompt_texts[i : i + self.api_batch_size]
            batch_results = asyncio.run(
                batch_send(
                    model_config=self.model_config,
                    prompts=batch_prompts,
                    system_prompt=system_prompt,
                )
            )
            api_responses.extend(batch_results)

        # Extract response text from API results
        response_texts = []
        for i, api_result in enumerate(api_responses):
            if api_result is None:
                logger.warning(f"API call failed for prompt {i}, using empty response")
                response_texts.append("")
            else:
                # Extract the content from the API response
                try:
                    content = api_result["choices"][0]["message"]["content"]
                    response_texts.append(content)
                except (KeyError, IndexError) as e:
                    logger.error(f"Error extracting content from API response {i}: {e}")
                    response_texts.append("")

        # Process responses to extract answers
        processed_answers = self._process_responses(response_texts)

        # Create output DataProto
        output = DataProto.from_dict(
            tensors={},
            non_tensors={
                "answers": np.array(processed_answers, dtype=object),
                "responses_str": np.array(response_texts, dtype=object),
            },
        )

        return output

    def _process_responses(self, response_texts: List[str]) -> List[str]:
        """
        Process API response texts to extract clean answers.

        Args:
            response_texts: List of raw response texts from API

        Returns:
            List of processed answer strings
        """
        processed = []

        for resp in response_texts:
            if self.config.cot or self.config.thinking:
                # findall returns a list of all inner matches
                matches = ANSWER_PATTERN.findall(resp)
                if matches:
                    # join them in case there are multiple <answer> blocks
                    answer = " ".join(m.strip() for m in matches)
                else:
                    # if no <answer> tags found, use empty string
                    answer = ""
            else:
                answer = resp

            # clean out unwanted chars
            answer = CLEAN_PATTERN.sub("", answer).strip()

            processed.append(answer)

        return processed


class APIActorRolloutWorker(Worker):
    """
    This worker uses API calls for actor rollout inference during evaluation.
    It generates questions/responses using external API instead of local model inference.

    Note: This is only for inference/evaluation, not for training.

    Example configuration:
        actor:
            model:
                api_model_name: "qwen/qwen3-30b-a3b-instruct-2507"
                temperature: 0.7
                top_p: 0.9
            api_batch_size: 200
    """

    def __init__(self, config: DictConfig, role: str = "actor_rollout", tokenizer=None):
        super().__init__()
        self.config = config
        self.role = role  # Store role for compatibility, though not used in API mode
        logger.info(f"APIActorRolloutWorker initialized with role: {role}")

        # Load tokenizer - either passed directly or from config path
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif hasattr(config.model, "tokenizer_path") and config.model.tokenizer_path:
            from verl.utils.fs import copy_to_local

            local_path = copy_to_local(config.model.tokenizer_path)
            trust_remote_code = config.model.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(
                local_path, trust_remote_code=trust_remote_code
            )
            logger.info(f"Loaded tokenizer from {config.model.tokenizer_path}")
        else:
            self.tokenizer = None
            logger.warning(
                "No tokenizer provided - API worker can only handle text input, not token IDs"
            )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        # Extract model configuration for API calls
        self.max_new_tokens = self.config.get("response_length", 50)

        self.model_config = {
            "name": self.config.model.get(
                "api_model_name", "qwen/qwen3-30b-a3b-instruct-2507"
            ),
            "temperature": self.config.model.get("temperature", None),
            "top_p": self.config.model.get("top_p", None),
            "max_tokens": self.max_new_tokens,  # Add max_tokens to model config
        }

        # API batch size for processing
        self.api_batch_size = self.config.get("api_batch_size", 200)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize API worker components."""
        logger.info(
            f"Initializing API Actor Worker with model: {self.model_config['name']}"
        )
        logger.info(f"API batch size: {self.api_batch_size}")
        logger.info(f"Max new tokens: {self.max_new_tokens}")
        logger.info(
            f"Temperature: {self.model_config['temperature']}, Top-p: {self.model_config['top_p']}"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_rollout(self):
        """Prepare for rollout."""
        logger.info("Starting API actor rollout")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def exit_rollout(self):
        """Clean up after rollout."""
        logger.info("Exiting API actor rollout")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_rollout_fanout(self, is_validate: bool):
        """
        Get the rollout fanout (number of parallel generations per prompt).

        Args:
            is_validate: Whether this is for validation or training

        Returns:
            Number of parallel rollouts to perform
        """
        if is_validate:
            return self.config.rollout.val_kwargs.n
        else:
            return self.config.rollout.n

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        Generate sequences using API calls for actor rollout.

        Can accept either:
        - Token IDs in prompts.batch['input_ids'] (decoded using tokenizer)
        - Text prompts in prompts.non_tensor_batch['prompt_text']

        Args:
            prompts: DataProto containing either token IDs or text prompts

        Returns:
            DataProto with generated responses
        """
        # Extract prompts - either from text or by decoding token IDs
        prompt_texts = prompts.non_tensor_batch.get("prompt_text")

        if prompt_texts is None:
            # Decode from input_ids
            if "input_ids" not in prompts.batch:
                raise ValueError(
                    "APIActorRolloutWorker requires either 'prompt_text' in non_tensor_batch "
                    "or 'input_ids' in batch"
                )
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is required when input_ids are provided. "
                    "Pass tokenizer to APIActorRolloutWorker.__init__()"
                )
            input_ids = prompts.batch["input_ids"]
            prompt_texts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

        batch_size = len(prompt_texts)

        # Extract system prompt if provided - handle both single string and list
        system_prompt = prompts.non_tensor_batch.get("system_prompt", None)
        if (
            system_prompt is not None
            and hasattr(system_prompt, "__len__")
            and not isinstance(system_prompt, str)
        ):
            # If it's a list/array, use first element (assuming all are same for actor)
            system_prompt = system_prompt[0] if len(system_prompt) > 0 else None

        # Make API calls in batches
        logger.info(f"Processing {batch_size} prompts via API for actor rollout")
        api_responses = []
        for i in range(0, batch_size, self.api_batch_size):
            batch_prompts = prompt_texts[i : i + self.api_batch_size]
            batch_results = asyncio.run(
                batch_send(
                    model_config=self.model_config,
                    prompts=batch_prompts,
                    system_prompt=system_prompt,
                )
            )
            api_responses.extend(batch_results)

        # Extract response text from API results
        response_texts = []
        for i, api_result in enumerate(api_responses):
            if api_result is None:
                logger.warning(f"API call failed for prompt {i}, using empty response")
                response_texts.append("")
            else:
                # Extract the content from the API response
                try:
                    content = api_result["choices"][0]["message"]["content"]
                    response_texts.append(content)
                except (KeyError, IndexError) as e:
                    logger.error(f"Error extracting content from API response {i}: {e}")
                    response_texts.append("")

        # For actor rollout, we need to return responses in a format compatible with the generation loop
        # The downstream code expects 'responses' as token IDs if available
        if self.tokenizer is not None:
            # Tokenize the full responses (not just the generated part)
            # This matches what vLLM/HF rollout returns
            tokenized = self.tokenizer(
                response_texts,
                padding="longest",
                truncation=True,
                max_length=self.max_new_tokens,
                return_tensors="pt",
                add_special_tokens=False,
            )

            response_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            output = DataProto.from_dict(
                tensors={
                    "responses": response_ids,
                    "attention_mask": attention_mask,
                    "response_length": torch.tensor(
                        [len(r.split()) for r in response_texts], dtype=torch.long
                    ),
                },
                non_tensors={"responses_str": np.array(response_texts, dtype=object)},
            )
        else:
            # Return only text responses if no tokenizer available
            output = DataProto.from_dict(
                tensors={},
                non_tensors={"responses_str": np.array(response_texts, dtype=object)},
            )

        return output
