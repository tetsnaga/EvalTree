import json
import argparse
import pandas as pd
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH-Hard", "MMLU-Pro", "BBH", "GPQA", ))
parser.add_argument("--description_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", "gpt-4o", ))
args = parser.parse_args()


with open("Datasets/{}/EvalTree/stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]_[stage4-CapabilityDescription-model={}].json".format(args.dataset, args.description_model), "r") as fin :
    TREE_DESCRIPTION = json.load(fin)

# Load dataset questions for leaf node display
df = pd.read_parquet("Datasets/{}/dataset.parquet".format(args.dataset))
if args.dataset == "MATH-Hard" :
    inputs = list(df["problem"])
elif args.dataset == "BBH" :
    inputs = list(df["input"])
elif args.dataset == "MMLU-Pro" :
    inputs = list(df["question"])
elif args.dataset == "GPQA" :
    inputs = []
    for _, row in df.iterrows() :
        q = row.get("Extra Revised Question") or row.get("Question")
        inputs.append(q)
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))

# Load model scores (only models with complete evaluations)
scores_df = pd.read_parquet("Datasets/{}/model_scores_no_nans.parquet".format(args.dataset))
model_cols = [c for c in scores_df.columns if c != "doc_hash"]
model2results = {model : list(scores_df[model].astype(int)) for model in model_cols}


def calculate_results(tree_description) :
    tree_results = {
        "size" : 0,
        "model2sum_metrics" : {},
    }
    if not isinstance(tree_description["subtrees"], int) :
        if isinstance(tree_description["subtrees"], list) :
            tree_results["subtrees"] = []
            for subtree_description in tree_description["subtrees"] :
                tree_results["subtrees"].append(calculate_results(subtree_description))
        elif isinstance(tree_description["subtrees"], dict) :
            tree_results["subtrees"] = {}
            for cluster, subtree_description in tree_description["subtrees"].items() :
                tree_results["subtrees"][cluster] = calculate_results(subtree_description)
        else :
            assert False

        tree_results["size"] = 0
        for model in model2results.keys() :
            tree_results["model2sum_metrics"][model] = 0
        for subtree_results in tree_results["subtrees"] if isinstance(tree_results["subtrees"], list) else tree_results["subtrees"].values() :
            tree_results["size"] += subtree_results["size"]
            for model, sum_metrics in subtree_results["model2sum_metrics"].items() :
                tree_results["model2sum_metrics"][model] += sum_metrics
    else :
        tree_results["size"] = 1
        for model, results in model2results.items() :
            tree_results["model2sum_metrics"][model] = results[tree_description["subtrees"]]
        tree_results["subtrees"] = tree_description["subtrees"]
    return tree_results

TREE_RESULTS = calculate_results(TREE_DESCRIPTION)


def merge(tree_results, tree_description, depth) :
    tree_data = {"capability" : tree_description["description"], "size" : 0, "depth" : depth}
    if not isinstance(tree_results["subtrees"], int) :
        tree_data["subtrees"] = []
        if isinstance(tree_results["subtrees"], list) :
            assert isinstance(tree_description["subtrees"], list)
            assert len(tree_results["subtrees"]) == len(tree_description["subtrees"])
            for subtree_results, subtree_description in zip(tree_results["subtrees"], tree_description["subtrees"]) :
                tree_data["subtrees"].append(merge(subtree_results, subtree_description, depth + 1))
        else :
            assert isinstance(tree_description["subtrees"], dict) and isinstance(tree_results["subtrees"], dict)
            assert set(tree_results["subtrees"].keys()) == set(tree_description["subtrees"].keys())
            for cluster, subtree_results in tree_results["subtrees"].items() :
                tree_data["subtrees"].append(merge(subtree_results, tree_description["subtrees"][cluster], depth + 1))
        for subtree_data in tree_data["subtrees"] :
            tree_data["size"] += subtree_data["size"]

        if "distinguishing" in tree_description :
            distinctions = tree_description["distinguishing"].split("\n")
            if len(distinctions) == len(tree_data["subtrees"]) :
                for distinction, subtree_data in zip(distinctions, tree_data["subtrees"]) :
                    subtree_data["distinction"] = distinction.strip()
    else :
        assert isinstance(tree_description["subtrees"], int)
        assert tree_results["subtrees"] == tree_description["subtrees"]
        tree_data["size"] = 1
        tree_data["input"] = inputs[tree_results["subtrees"]]
        if isinstance(tree_data["input"], str) and len(tree_data["input"]) > 384 :
            tree_data["input"] = tree_data["input"][: 384] + " ..."
        tree_data["subtrees"] = tree_results["subtrees"]

    assert tree_results["size"] == tree_data["size"]

    tree_data["ranking"] = [[model, sum_metrics / tree_results["size"]] for model, sum_metrics in tree_results["model2sum_metrics"].items()]

    if tree_data["size"] >= 5 :
        tree_data["CI"] = {}
        for model, sum_metrics in tree_results["model2sum_metrics"].items() :
            lower_bound, upper_bound = sm.stats.proportion_confint(sum_metrics, tree_results["size"], alpha = 0.05, method = "beta")
            tree_data["CI"][model] = [lower_bound, upper_bound]

    tree_data["ranking"].sort(key = lambda x : x[1], reverse = True)
    return tree_data


import os
os.makedirs("data", exist_ok = True)
with open("data/{}.json".format(args.dataset), "w") as fout :
    json.dump(merge(TREE_RESULTS, TREE_DESCRIPTION, depth = 1), fout, indent = 2)
print("Saved to data/{}.json".format(args.dataset))
