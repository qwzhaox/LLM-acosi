import time
import asyncio

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from tqdm import tqdm
from dotenv import load_dotenv

from utils import alpaca_format_prompt

ENV_PATH = "config/.env"

load_dotenv(ENV_PATH)


def get_formatted_example_prompts(examples, formatted_prompt, intro_blurb):
    formatted_example_prompts = []
    for ex in examples:
        formatted_example_prompts.extend(
            [
                SystemMessage(content=intro_blurb),
                HumanMessage(content=formatted_prompt.format(instruction=ex[0])),
                AIMessage(content=ex[1]),
            ]
        )
    return formatted_example_prompts


def query_gpt(
    prompts,
    examples,
    deployment_name,
    max_tokens=1024,
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
        num_prompts = len(prompts)

    print(f"{num_prompts} examples.")
    abbreviated_prompts = prompts[:num_prompts]

    formatted_prompt, _, intro_blurb = alpaca_format_prompt()
    formatted_example_prompts = get_formatted_example_prompts(
        examples, formatted_prompt, intro_blurb
    )

    def invoke_serially(llm, prompts):
        import pdb

        pdb.set_trace()
        return [
            llm.invoke(
                formatted_example_prompts
                + [
                    SystemMessage(content=intro_blurb),
                    HumanMessage(content=formatted_prompt.format(instruction=prompt)),
                ]
            )
            for prompt in tqdm(prompts)
        ]

    # async def async_invoke(llm, message):
    #     resp = await llm.ainvoke([message])
    #     return resp.content

    async def invoke_concurrently_batch(llm, prompts):
        resp = await llm.abatch(
            [
                formatted_example_prompts
                + [
                    SystemMessage(content=intro_blurb),
                    HumanMessage(content=formatted_prompt.format(instruction=prompt)),
                ]
                for prompt in prompts
            ],
            {"max_concurrency": 5},
        )
        filtered = [rev.content for rev in resp]
        return filtered

    if serial:
        s = time.perf_counter()
        output = invoke_serially(llm, abbreviated_prompts)
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Serial executed {num_prompts} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    if parallel:
        s = time.perf_counter()
        # output = asyncio.run(invoke_concurrently(llm, abbreviated_exs))
        output = asyncio.run(invoke_concurrently_batch(llm, abbreviated_prompts))
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Concurrent executed {num_prompts} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    return output
