import os
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "Chatbot-Arena", "Chatbot-Arena_NEW", "MMLU", "DS-1000", "MATH-Hard", "MMLU-Pro", "BBH", "GPQA", ))
parser.add_argument("--tree_path", type = str, default = "stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]")
parser.add_argument("--description_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", "gpt-4o", ))
parser.add_argument("--distinguishing_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", "gpt-4o", ))
parser.add_argument("--num_procs", type = int, default = 64)
args = parser.parse_args()

with open(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}_[stage4-CapabilityDescription-model={}].json".format(args.tree_path, args.description_model)), "r") as fin :
    TREE_DESCRIPTION = json.load(fin)

with open("EvalTree/stage5-CapabilityDistinguishing/prompt.txt", "r") as fin :
    PROMPT = fin.read()
OPENAI_KWARGS = {
    "model" : args.distinguishing_model,
    "max_tokens" : 4096,
    "temperature" : 0.0,
    "seed" : 0,
}


PROMPTS = []
def prepare_prompts(tree_description) :
    if isinstance(tree_description["subtrees"], int) :
        return
    sub_capabilities = []
    for subtree_description in tree_description["subtrees"] if isinstance(tree_description["subtrees"], list) else tree_description["subtrees"].values() :
        prepare_prompts(subtree_description)
        sub_capabilities.append(subtree_description["description"])
    prompt = PROMPT.format(high_level_capability = tree_description["description"], sub_capabilities = "\n".join(sub_capabilities), num_sub_capabilities = len(sub_capabilities))
    tree_description["distinguishing"] = prompt
    PROMPTS.append(prompt)
prepare_prompts(TREE_DESCRIPTION)


def Process(prompt) :
    chatml = prompt_to_chatml(prompt)
    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_completion(client, chatml, OPENAI_KWARGS)
with concurrent.futures.ThreadPoolExecutor(max_workers = args.num_procs) as executor :
    outputs = list(tqdm(executor.map(Process, PROMPTS), desc = "dataset", total = len(PROMPTS)))
print("cost = {}".format(sum([output["cost"] for output in outputs])))
outputs = {prompt : output["response"] for prompt, output in zip(PROMPTS, outputs)}


def fill_responses(tree_description) :
    if isinstance(tree_description["subtrees"], int) :
        return
    tree_description["distinguishing"] = outputs[tree_description["distinguishing"]]
    for subtree_description in tree_description["subtrees"] if isinstance(tree_description["subtrees"], list) else tree_description["subtrees"].values() :
        fill_responses(subtree_description)
fill_responses(TREE_DESCRIPTION)


with open(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}_[stage4-CapabilityDescription-model={}].json".format(args.tree_path, args.description_model)), "w") as fout :
    json.dump(TREE_DESCRIPTION, fout, indent = 2)
