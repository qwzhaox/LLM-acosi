import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from nltk import word_tokenize
from utils import flatten_output

from transformers.generation import GenerationConfig

device = 0 if torch.cuda.is_available() else -1

ASPECT_IDX = 0
CATEGORY_IDX = 1
SENTIMENT_IDX = 2
OPINION_IDX = 3
IMPLICIT_INDICATOR_IDX = 4


def alpaca_format_prompt():
    instruction_key = "### Instruction:"
    response_key = "### Response:"
    intro_blurb = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    prompt_for_generation_format = """{intro}
                                      {instruction_key}
                                      {instruction}
                                      {response_key}
                                    """.format(
        intro=intro_blurb,
        instruction_key=instruction_key,
        instruction="{instruction}",
        response_key=response_key,
    )
    return prompt_for_generation_format, response_key


def get_formatted_annotations(annotations):
    annots = []
    for annotation in annotations:
        if len(annotation) == 4:
            new_annot_str = (
                "(Aspect: {}, Category: {}, Sentiment: {}, Opinion: {})".format(
                    annotation[ASPECT_IDX],
                    annotation[CATEGORY_IDX],
                    annotation[SENTIMENT_IDX],
                    annotation[OPINION_IDX],
                )
            )
        elif len(annotation) == 5:
            new_annot_str = "(Aspect: {}, Category: {}, Sentiment: {}, Opinion: {}, Implicit/Explicit: {})".format(
                annotation[ASPECT_IDX],
                annotation[CATEGORY_IDX],
                annotation[SENTIMENT_IDX],
                annotation[OPINION_IDX],
                annotation[IMPLICIT_INDICATOR_IDX],
            )

        annots.append(new_annot_str)

    annots_str = "[" + ", ".join(annots) + "]"

    return annots_str


def run_pipeline(args, prompt, examples=[], absa_task="extract-acosi"):
    # Initialize the pipeline with the specified model, and set the device
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, model_max_length=1024
    )
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    gen_config = GenerationConfig.from_pretrained(args.model_name)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.max_length = 1024
    # pre_config = PretrainedConfig.from_pretrained(args.model_name)

    print("Initializing pipeline...")

    model_pipe = pipeline(
        args.task,
        model=args.model_name,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=args.remote,
        # config=pre_config,
    )

    print("Loading dataset...")

    with open(args.dataset_file, "r") as f:
        dataset = f.readlines()

    formatted_prompt, response_key = alpaca_format_prompt()

    print("Processing dataset...")

    prompts = []

    for i, data in enumerate(tqdm(dataset, desc="Processing", unit="item")):
        review = data.split("####")[0]
        annotations = eval(data.split("####")[1])

        review_str = f"Review: {review.strip()}\n"
        annotations_str = ""
        if absa_task == "acos-extend":
            annotations_str = (
                f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
            )

        examples_str = "".join(examples)
        bare_prompt = (
            prompt + examples_str + f"Your Task {i}:\n" + review_str + annotations_str
        )
        final_prompt = formatted_prompt.format(instruction=bare_prompt)
        prompts.append(final_prompt)

    print("Running pipeline...")

    output = model_pipe(prompts, generation_config=gen_config)

    flat_output = flatten_output(output)
    total_out_tokens = 0
    for out in flat_output:
        total_out_tokens += len(word_tokenize(out["generated_text"].strip()))
    print(f"Total output tokens: {total_out_tokens}")
    print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")

    return output, response_key
