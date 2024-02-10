import time
import asyncio

from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from tqdm import tqdm
from dotenv import load_dotenv

from utils import (
    XU_ETAL_INPUT_KEY,
    XU_ETAL_OUTPUT_KEY,
    OLD_REVIEW_KEY,
    OLD_RESPONSE_KEY,
    OLD_INTRO_BLURB,
    EXAMPLE_REVIEW, 
    EXAMPLE_RESPONSE
)

ENV_PATH = "config/.env"

load_dotenv(ENV_PATH)


def get_gpt_prompt_instr(prompt_instr, is_old_prompt=False):
    gpt_prompt_instr = [SystemMessage(content=OLD_INTRO_BLURB)] if is_old_prompt else []
    gpt_prompt_instr.extend([
            HumanMessage(content=prompt_instr["instruction"]),
            HumanMessage(content=prompt_instr["context"]),
            HumanMessage(content=prompt_instr["output-format"]),
    ])
    return gpt_prompt_instr


def get_formatted_examples(examples, review_key, response_key, gpt_prompt_instr=None, is_old_prompt=False):
    formatted_examples = []
    for ex in examples:
        formatted_example = gpt_prompt_instr if is_old_prompt else []
        formatted_example.extend([
            HumanMessage(content=f"{review_key} {ex[EXAMPLE_REVIEW]}"),
            HumanMessage(content=response_key),
            AIMessage(content=ex[EXAMPLE_RESPONSE]),
        ])
        
        formatted_examples.extend(formatted_example)

    return formatted_examples


def get_gpt_prompts(abbreviated_prompts, gpt_prompt_instr, review_key, response_key, is_old_prompt=False):
    gpt_prompts = []

    for prompt in abbreviated_prompts:
        examples = get_formatted_examples(prompt["examples"], review_key, response_key, gpt_prompt_instr=gpt_prompt_instr, is_old_prompt=is_old_prompt)
        gpt_prompt = examples + gpt_prompt_instr if is_old_prompt else gpt_prompt_instr + examples
        gpt_prompt.extend([
            HumanMessage(content=f"{review_key} {prompt['review']}"),
            HumanMessage(content=response_key),
        ])
        
        gpt_prompts.append(gpt_prompt)
    
    return gpt_prompts


def query_gpt(
    deployment_name,
    prompts,
    max_tokens=1024,
    is_old_prompt=False,
    serial=False,
    parallel=True,
    num_prompts=None,
):
    llm = AzureChatOpenAI(
        openai_api_version="2023-07-01-preview",
        azure_deployment=deployment_name,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=7,
    )

    if not num_prompts:
        num_prompts = len(prompts) - 1

    print(f"{num_prompts} examples.")

    abbreviated_prompts = prompts[1:num_prompts+1]
    prompt_instr = prompts[0]

    review_key = OLD_REVIEW_KEY if is_old_prompt else XU_ETAL_INPUT_KEY
    response_key = OLD_RESPONSE_KEY if is_old_prompt else XU_ETAL_OUTPUT_KEY

    gpt_prompt_instr = get_gpt_prompt_instr(prompt_instr, is_old_prompt=is_old_prompt)
    gpt_prompts = get_gpt_prompts(abbreviated_prompts, gpt_prompt_instr, review_key, response_key, is_old_prompt=is_old_prompt)

    def invoke_serially(llm, prompts):
        import pdb

        pdb.set_trace()
        return [
            llm.invoke(prompt)
            for prompt in tqdm(prompts, desc="Serially invoking GPT-3", unit="prompt")
        ]

    # async def async_invoke(llm, message):
    #     resp = await llm.ainvoke([message])
    #     return resp.content

    async def invoke_concurrently_batch(llm, prompts):
        resp = await llm.abatch(
            [
                prompt
                for prompt in tqdm(prompts, desc="Concurrently invoking GPT-3", unit="prompts")
            ],
            {"max_concurrency": 5},
        )
        filtered = [rev.content for rev in resp]
        return filtered

    if serial:
        s = time.perf_counter()
        output = invoke_serially(llm, gpt_prompts)
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Serial executed {num_prompts} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    if parallel:
        s = time.perf_counter()
        # output = asyncio.run(invoke_concurrently(llm, abbreviated_exs))
        output = asyncio.run(invoke_concurrently_batch(llm, gpt_prompts))
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Concurrent executed {num_prompts} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    return output
