import torch
from json import load
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from nltk import word_tokenize
from transformers.generation import GenerationConfig
from pathlib import Path
from random import sample, seed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    get_dataset_domain,
    xu_etal_format_prompt,
    get_xu_etal_formatted_annotations,
    old_format_prompt,
    get_old_formatted_annotations,
    flatten_output,
    XU_ETAL_INPUT_KEY,
    XU_ETAL_OUTPUT_KEY,
    OLD_REVIEW_KEY,
    OLD_RESPONSE_KEY,
    EXAMPLE_REVIEW,
    EXAMPLE_RESPONSE,
)
from gpt_pipeline import query_gpt
from prompts import PROMPTS, PROMPTS_OLD, PROMPTS_COMBO, CATE_DICT

device = 0 if torch.cuda.is_available() else -1
seed(0)

class Pipeline:
    def __init__(self, args):
        self.model = args.model_name
        self.tokenizer_name = args.tokenizer_name
        self.task = args.task

        self.max_length = args.max_length
        self.max_new_tokens = args.max_new_tokens
        self.remote = args.remote

        self.absa_task = args.absa_task
        self.dataset_file = args.dataset_file
        self.annotation_source = args.annotation_source

        self.k_examples = args.k_examples
        self.selection_method = args.selection_method
        self.limit = args.limit

        self.is_old_prompt = args.is_old_prompt
        self.is_combo_prompt = args.is_combo_prompt
        

    def get_model_output(self):
        prompts, reviews = self.get_prompts()
        if "gpt" in self.model.lower():
            output = query_gpt(
                self.model.lower(),
                prompts,
                max_tokens=self.max_new_tokens,
                is_old_prompt=self.is_old_prompt,
            )
        else:
            output = self.run_pipeline(prompts)

        return output, reviews

    def get_prompts(self):
        with open(self.dataset_file, "r") as f:
            dataset = f.readlines()

        dataset_path = Path(self.dataset_file).parent
        dataset_domain = get_dataset_domain(dataset_path)

        instruction, context, output_format = self.__get_prompt_info(dataset_domain)
        formatted_prompt = self.__get_format_prompt(instruction, context, output_format)

        print("Processing dataset...")

        prompts = []
        if "gpt" in self.model:
            prompt_instr = {
                    "instruction": instruction,
                    "context": context,
                    "output-format": output_format,
                    }
            prompts.append(prompt_instr)

        examples = []
        if self.is_old_prompt:
            examples = self.__get_examples_old(dataset_path.name, self.absa_task)

        train_dataset, acos_extend_train_dataset = self.__get_train_dataset(dataset_path)
        
        reviews = []
        outside_annotations = []
        if ("mvp" in self.annotation_source) or (self.annotation_source == "gen-scl-nat"):
            with open(f"model_output/supervised/{self.annotation_source}/pred.json", "r") as file:
                outside_annotations = load(file)

        for i, data in enumerate(tqdm(dataset, desc="Processing Dataset", unit="item")):
            review = data.split("####")[0]
            reviews.append(review)

            if self.annotation_source == "true":
                annotations = eval(data.split("####")[1])
            elif ("mvp" in self.annotation_source) or (self.annotation_source == "gen-scl-nat"):
                annotations = outside_annotations[i]
            else:
                raise NotImplementedError("Invalid annotation source.")
            
            review_str = self.__get_review_str(review, annotations)

            if not self.is_old_prompt:
                examples = self.__get_examples(review, train_dataset, acos_extend_train_dataset)

            if "gpt" in self.model:
                prompt = {
                    "examples": examples,
                    "review": review_str,
                }
            else:
                prompt = self.__format_llama_prompt(formatted_prompt, review_str, examples)

            prompts.append(prompt)

        return prompts, reviews

    def run_pipeline(self, prompts):
        # Initialize the pipeline with the specified model, and set the device
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, model_max_length=self.max_length
        )
        # model = AutoModelForCausalLM.from_pretrained(args.model_name)
        gen_config = GenerationConfig.from_pretrained(self.model)
        gen_config.max_new_tokens = self.max_new_tokens
        gen_config.max_length = self.max_length
        # pre_config = PretrainedConfig.from_pretrained(args.model_name)

        print("Initializing pipeline...")

        model_pipe = pipeline(
            self.task,
            model=self.model,
            tokenizer=tokenizer,
            device_map="auto",
            trust_remote_code=self.remote,
            # config=pre_config,
        )

        print("Running pipeline...")

        output = model_pipe(prompts, generation_config=gen_config)

        flat_output = flatten_output(output)
        total_out_tokens = 0
        for out in flat_output:
            total_out_tokens += len(word_tokenize(out["generated_text"].strip()))
        print(f"Total output tokens: {total_out_tokens}")
        print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")

        return output

    ### HELPERS ###
    def __get_prompt_info(self, dataset_domain):
        if self.is_combo_prompt:
            prompt_dict = PROMPTS_COMBO
        elif self.is_old_prompt:
            prompt_dict = PROMPTS_OLD
        else:
            prompt_dict = PROMPTS

        instruction = prompt_dict[self.absa_task]["instruction"]
        context = prompt_dict[self.absa_task]["context"].format(category_list=CATE_DICT[dataset_domain])
        output_format = prompt_dict[self.absa_task]["output-format"]

        return instruction, context, output_format

    def __get_format_prompt(self, instruction, context, output_format):
        if self.is_old_prompt:
            formatted_prompt = old_format_prompt(instruction, context, output_format)
        else:
            formatted_prompt = xu_etal_format_prompt(instruction, context, output_format)

        return formatted_prompt

    def __get_review_str(self, review, annotations):
        review_str = f"{review.strip()}\n"
        annotations_str = ""
        if self.absa_task == "acos-extend":
            if self.is_old_prompt or self.is_combo_prompt:
                annotations_str = f"ACOS quadruples: {get_old_formatted_annotations(annotations)}\n"
            else:
                annotations_str = f"ACOS quadruples: {get_xu_etal_formatted_annotations(annotations)}\n"

        review_str = review_str + annotations_str
        return review_str

    def __select_similar_reviews(self, review, train_dataset):
        train_reviews = [example.split("####")[0] for example in train_dataset]
        if self.selection_method == "tf-idf":
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(train_reviews)
            review_vector = vectorizer.transform([review])
            cosine_similarities = cosine_similarity(review_vector, X).flatten()
            indices = cosine_similarities.argsort()[-self.k_examples:][::-1]
            return indices
        else:
            raise NotImplementedError("Only tf-idf selection method is supported.")

    def __get_train_dataset(self, dataset_path):
        train_dataset = []
        with open(Path(dataset_path / "train.txt"), "r") as f:
                train_dataset = f.readlines()

        acos_extend_train_dataset_path = Path(str(dataset_path).replace('acos', 'acosi'))
        acos_extend_train_dataset = []
        if self.absa_task == "acos-extend":
            with open(Path(acos_extend_train_dataset_path / "train.txt"), "r") as f:
                acos_extend_train_dataset = f.readlines()

        if self.limit:
            limit_indices = sample(range(len(train_dataset)), k=self.limit)
            train_dataset = [train_dataset[i] for i in limit_indices]
            if acos_extend_train_dataset:
                acos_extend_train_dataset = [acos_extend_train_dataset[i] for i in limit_indices]
            assert len(train_dataset) == self.limit

        return train_dataset, acos_extend_train_dataset

    def __get_examples(self, review, train_dataset, acos_extend_train_dataset=[]):
        if self.limit:
            assert len(train_dataset) == self.limit

        examples = []

        if self.selection_method == "random":
            indices = sample(range(len(train_dataset)), k=self.k_examples)
        elif self.selection_method == "tf-idf":
            indices = self.__select_similar_reviews(review, train_dataset)

        for idx in indices:
            review = train_dataset[idx].split("####")[0]
            annotations = eval(train_dataset[idx].split("####")[1])
            if self.is_combo_prompt:
                annotations = get_old_formatted_annotations(annotations)
            else:
                annotations = get_xu_etal_formatted_annotations(annotations)

            if self.absa_task == "acos-extend":
                review = f"{review}\nACOS quadruples: {annotations}"
                annotations = eval(acos_extend_train_dataset[idx].split("####")[1])
                if self.is_combo_prompt:
                    response = get_old_formatted_annotations(annotations, opinion_span_only=True)
                else:
                    response = get_xu_etal_formatted_annotations(annotations, opinion_span_only=True)
            else:
                response = annotations
            
            examples.append((review, response))
        
        return examples

    def __get_examples_old(self, dataset):
        examples = []

        for example in PROMPTS_OLD[self.absa_task]["examples"][dataset]:
            if self.absa_task == "acos-extend":
                review = f"{example['review']}\nACOS quadruples: {example['acos-quadruples']}"
                response = f"{example['response']}\n{example['response-explanation']}"
            else:
                review = example["review"]
                response = example["response"]

            examples.append((review, response))

        return examples

    def __format_llama_prompt(self, formatted_prompt, review_str, examples):
        example_str = ""
        for example in examples:
            if self.is_old_prompt:
                example_str += formatted_prompt
                review_key = OLD_REVIEW_KEY
                response_key = OLD_RESPONSE_KEY
            else:
                review_key = XU_ETAL_INPUT_KEY
                response_key = XU_ETAL_OUTPUT_KEY

            example_str += f"{review_key} {example[EXAMPLE_REVIEW]}\n"
            example_str += f"{response_key} {example[EXAMPLE_RESPONSE]}\n"

        if self.is_old_prompt:
            prompt = f"{example_str}{formatted_prompt}{OLD_REVIEW_KEY}{review_str}{OLD_RESPONSE_KEY}"
        else:
            prompt = f"{formatted_prompt}{example_str}{XU_ETAL_INPUT_KEY}{review_str}{XU_ETAL_OUTPUT_KEY}"

        return prompt
