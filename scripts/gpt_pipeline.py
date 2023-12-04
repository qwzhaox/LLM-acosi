import os
import re

import argparse
from datetime import datetime

# from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import time
import asyncio

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
from dotenv import load_dotenv

env_path = "config/.env"

load_dotenv(env_path)


def comparison_fn(
    exs,
    prefix,
    sys_prefix,
    deployment_name,
    serial=False,
    parallel=True,
    num_examples=None,
):
    llm = AzureChatOpenAI(
        openai_api_version="2023-07-01-preview",
        azure_deployment=deployment_name,
        temperature=0.0,
        max_tokens=400,
        max_retries=7,
    )

    if not num_examples:
        num_examples = len(exs)

    print(f"{num_examples} examples.")
    abbreviated_exs = exs[:num_examples]

    def invoke_serially(llm, exs):
        import pdb

        pdb.set_trace()
        return [
            llm.invoke(
                [
                    SystemMessage(content=sys_prefix),
                    HumanMessage(content=prefix + query),
                ]
            )
            for query in tqdm(exs)
        ]

    # async def async_invoke(llm, message):
    #     resp = await llm.ainvoke([message])
    #     return resp.content

    async def invoke_concurrently_batch(llm, exs):
        resp = await llm.abatch(
            [
                [
                    SystemMessage(content=sys_prefix),
                    HumanMessage(content=prefix + query),
                ]
                for query in exs
            ],
            {"max_concurrency": 5},
        )
        filtered = [ex.content for ex in resp]
        return filtered

    if serial:
        s = time.perf_counter()
        output = invoke_serially(llm, abbreviated_exs)
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Serial executed {num_examples} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    if parallel:
        s = time.perf_counter()
        # output = asyncio.run(invoke_concurrently(llm, abbreviated_exs))
        output = asyncio.run(invoke_concurrently_batch(llm, abbreviated_exs))
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Concurrent executed {num_examples} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    return output


def main(config):
    """
    Trains the model

    :param config:
    :return:
    """

    if config.model_architecture == "OpenAI":
        replace_list = ["</s>", "<s>", "<pad>"]
        tokenizer = AutoTokenizer.from_pretrained("allenai/PRIMERA")
        dataset_reader = get_dataset_reader(config)
        datamodule = FinetuneDataModule(config, tokenizer, dataset_reader)

        dataset = dataset_reader.get_full_dataset(tokenizer)

        _val_inputs = [
            tokenizer.decode(elt["input_ids"], skip_special_tokens=False)
            for elt in dataset["val"]
        ]
        cleaned_val_inputs = [
            re.sub(r"|".join(map(re.escape, replace_list)), "", elt)
            for elt in _val_inputs
        ]
        val_gold = [
            tokenizer.decode(elt["output_ids"], skip_special_tokens=True)
            for elt in dataset["val"]
        ]

        _test_inputs = [
            tokenizer.decode(elt["input_ids"], skip_special_tokens=False)
            for elt in dataset["test"]
        ]
        test_gold = [
            tokenizer.decode(elt["output_ids"], skip_special_tokens=True)
            for elt in dataset["test"]
        ]
        cleaned_test_inputs = [
            re.sub(r"|".join(map(re.escape, replace_list)), "", elt)
            for elt in _test_inputs
        ]

        # val_pred = [get_chatgpt_results(elt, config.input_prefix) for elt in cleaned_val_inputs]
        # test_pred = [get_chatgpt_results(elt, config.input_prefix) for elt in cleaned_test_inputs]

        # JOE -> WENZHAO: cleaned_val_inputs is a list of examples
        pred = comparison_fn(
            cleaned_val_inputs,
            config.input_prefix,
            config.sys_prefix,
            config.deployment_name,
        )
