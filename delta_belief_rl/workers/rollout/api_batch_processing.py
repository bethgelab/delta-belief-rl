import json
import httpx
import asyncio

from delta_belief_rl.workers.rollout.api_key import openrouter_key


def batch_process_prompts(
    model_config: dict, prompts: list[str], batchsize: int = 200
) -> list[any]:
    results = []
    for i in range(0, len(prompts), batchsize):
        batch_prompts = prompts[i : i + batchsize]
        res = asyncio.run(batch_send(model_config, batch_prompts))
        results.extend(res)

    return results


async def fetch_response(model, prompt, system_prompt=None, effort=None, max_retries=5):
    """
    Make an async API request to OpenRouter for translation.

    Implements exponential backoff retry logic for handling timeouts and failures.

    Args:
        session: HTTP session (unused, kept for interface compatibility).
        model (str): Model identifier (e.g., 'qwen/qwen3-235b-a22b-2507').
        prompt (str): The text to translate.
        effort (str|None): Reasoning effort level ('high', 'medium', 'low') or None.
        max_retries (int): Maximum number of retry attempts (default: 5).

    Returns:
        api_response_json

    Raises:
        httpx.ReadTimeout: If all retry attempts fail.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model["name"],
        "temperature": model["temperature"],
        "top_p": model["top_p"],
        "provider": {
            "order": [
                "z-ai",
                "mistral",
                "atlas-cloud",
                "novita",
                "deepinfra",
                "wandb",
                "google-vertex",
            ],
            "allow_fallbacks": True,
        },
        "reasoning": {"effort": effort, "exclude": False},
    }

    # Add max_tokens if specified in model config
    if "max_tokens" in model:
        payload["max_tokens"] = model["max_tokens"]
    payload["messages"] = []
    if system_prompt is not None:
        payload["messages"] = [{"role": "system", "content": system_prompt}]
    payload["messages"].append({"role": "user", "content": prompt})

    timeout = httpx.Timeout(
        600.0, connect=60.0
    )  # 10 min read timeout, 1 min connect timeout

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url, headers=headers, data=json.dumps(payload)
                )
                result = response.json()
            return result
        except httpx.ReadTimeout:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                print(
                    f"Timeout for prompt '{prompt[:50]}...', retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
            else:
                print(
                    f"Failed after {max_retries} attempts for prompt: '{prompt[:50]}...'"
                )
                raise


async def _return_none(p):
    """Helper coroutine that returns a None result for a prompt."""
    return None


async def batch_send(model_config, prompts, system_prompt=None):
    """
    Translate a batch of prompts from source to target language.

    Args:
        model_config (dict): Model configuration with 'name' and 'effort' keys.
        prompts (list[str]): List of texts to translate.
        reasoning_effort (str|None): Reasoning effort level or None (unused, effort comes from model dict).

    Returns:
        list[tuple]: List of (original_prompt, api_response) tuples.

    Raises:
        Exception: If any translation fails after retries.
    """

    tasks = [
        fetch_response(model_config, p, system_prompt)
        if p is not None
        else _return_none(p)
        for p in prompts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions in results
    for i, result in enumerate(results):
        if result is None or isinstance(result, Exception):
            print(f"Error processing prompt {i}: {result}")
            results[i] = None  # Mark as failed

    return results

