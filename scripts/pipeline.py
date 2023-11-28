import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from nltk import word_tokenize
from utils import flatten_output, EXAMPLE_REVIEW, EXAMPLE_RESPONSE

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
            # new_annot_str = f"(Aspect: {annotation[ASPECT_IDX]}, Category: {annotation[CATEGORY_IDX]}, Sentiment: {annotation[SENTIMENT_IDX]}, Opinion: {annotation[OPINION_IDX]})"
            new_annot_str = f"[A] {annotation[ASPECT_IDX]} [C] {annotation[CATEGORY_IDX]} [S] {annotation[SENTIMENT_IDX]} [O] {annotation[OPINION_IDX]}"
        elif len(annotation) == 5:
            new_annot_str = f"[A] {annotation[ASPECT_IDX]} [C] {annotation[CATEGORY_IDX]} [S] {annotation[SENTIMENT_IDX]} [O] {annotation[OPINION_IDX]} [I] {annotation[IMPLICIT_INDICATOR_IDX]}"

        annots.append(new_annot_str)

    annots_str = " [SSEP] ".join(annots) + " [END]"

    return annots_str


def run_pipeline(args, prompt, examples=[], absa_task="extract-acosi"):
    # Initialize the pipeline with the specified model, and set the device
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, model_max_length=args.max_length
    )
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    gen_config = GenerationConfig.from_pretrained(args.model_name)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.max_length = args.max_length
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

    for data in tqdm(dataset, desc="Processing", unit="item"):
        review = data.split("####")[0]
        annotations = eval(data.split("####")[1])

        review_str = f"Review: {review.strip()}\n"
        annotations_str = ""
        if absa_task == "acos-extend":
            annotations_str = (
                f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
            )

        example_prompts = []
        for example in examples:
            bare_example_prompt = prompt + example[EXAMPLE_REVIEW]
            example_prompt = formatted_prompt.format(instruction=bare_example_prompt)
            complete_example = example_prompt + example[EXAMPLE_RESPONSE]
            example_prompts.append(complete_example)

        example_str = "\n".join(example_prompts) + "\n"
        bare_prompt = prompt + review_str + annotations_str
        final_prompt = example_str + formatted_prompt.format(instruction=bare_prompt)
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
