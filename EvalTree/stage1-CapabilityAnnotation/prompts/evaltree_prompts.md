# EvalTree Prompts for New Benchmarks

Custom prompts for the **Capability Annotation** (Stage 1) and **Capability Description** (Stage 4) stages of EvalTree's tree construction pipeline, adapted for four benchmarks: BIG-Bench Hard (BBH), GPQA, MATH Hard, and MMLU-Pro.

Prompts follow the patterns established in Appendix A of the EvalTree paper (Tables 1–8). All prompts use `max_new_tokens=1024` and `temperature=0.0` by default, consistent with the paper's configuration.

---

## 1. BIG-Bench Hard (BBH)

BBH is a suite of 23 challenging tasks from BIG-Bench where prior language models did not outperform average human raters. Tasks span algorithmic, logical, commonsense, and multi-step reasoning, with diverse output formats (boolean values, multiple-choice answers, sequences, etc.).

### Stage 1: Capability Annotation

**System Prompt**

```
Given a challenging reasoning task with a question and its correct answer (including a chain-of-thought explanation if available), generate a gerund phrase that thoroughly and precisely describes the **specific** reasoning skill or capability required to arrive at the correct answer.
```

**User Prompt**

```
## Question
{input}

## Answer
{output}

## Requirement
- The skill description should be an action-oriented gerund phrase that is **informative** and **detailed**.
- The phrase should refer to a **specific** skill or capability that comprehensively covers the key aspects of the reasoning required, without including any context or specifics from the question or answer.
- Avoid unnecessary elements unrelated to the core capability.
- Please output **only a gerund phrase** describing the skill, with NO additional text.
```

### Stage 4: Capability Description

**System Prompt**

```
Given a set of phrases, each summarizing the reasoning skills or capabilities needed to solve challenging reasoning tasks within a specific group, generate a gerund phrase that summarizes the collective set of reasoning skills or capabilities described across all groups.
```

**User Prompt**

```
## Task
You are given a set of phrases, each summarizing the reasoning skills or capabilities needed to solve challenging reasoning tasks within a specific group. There are {group_number} groups in total. Your task is to **summarize** the collective set of reasoning skills or capabilities that represents the union of these descriptions in a detailed and informative manner.

## Skill Descriptions
{skill_descriptions}

## Requirements
- The output should be a **single gerund phrase** that succinctly summarizes the overarching reasoning skill or capability represented by the union of all the provided phrases.
- The output should comprehensively cover each skill description without going beyond them.
- The output should not simply enumerate the given phrases but instead provide a meaningful and informative summary of the reasoning skills or capabilities they collectively represent.
- Please output **only a gerund phrase** summarizing the reasoning skill or capability, with NO additional text.
```

---

## 2. GPQA (Graduate-Level Google-Proof Q&A)

GPQA is a dataset of 448 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry. Questions are designed to be extremely difficult, requiring deep domain expertise that cannot be resolved by web search alone. Each question has four answer choices.

### Stage 1: Capability Annotation

**System Prompt**

```
Given a graduate-level science question with multiple answer choices and an explanation of the correct answer, generate a gerund phrase that thoroughly and precisely describes the **specific** scientific reasoning skill or domain expertise required to determine the correct answer.
```

**User Prompt**

```
## Question
{input}

## Correct Answer and Explanation
{output}

## Requirement
- The skill description should be an action-oriented gerund phrase that is **informative** and **detailed**.
- The phrase should refer to a **specific** skill or capability that comprehensively covers the key aspects of the scientific reasoning and domain knowledge required, without including any context or specifics from the question or answer.
- Avoid unnecessary elements unrelated to the core capability.
- Please output **only a gerund phrase** describing the skill, with NO additional text.
```

### Stage 4: Capability Description

**System Prompt**

```
Given a set of phrases, each summarizing the scientific reasoning skills or domain expertise needed to answer graduate-level science questions within a specific group, generate a gerund phrase that summarizes the collective set of scientific skills or capabilities described across all groups.
```

**User Prompt**

```
## Task
You are given a set of phrases, each summarizing the scientific reasoning skills or domain expertise needed to answer graduate-level science questions within a specific group. There are {group_number} groups in total. Your task is to **summarize** the collective set of scientific skills or capabilities that represents the union of these descriptions in a detailed and informative manner.

## Skill Descriptions
{skill_descriptions}

## Requirements
- The output should be a **single gerund phrase** that succinctly summarizes the overarching scientific skill or capability represented by the union of all the provided phrases.
- The output should comprehensively cover each skill description without going beyond them.
- The output should not simply enumerate the given phrases but instead provide a meaningful and informative summary of the scientific skills or capabilities they collectively represent.
- Please output **only a gerund phrase** summarizing the scientific skill or capability, with NO additional text.
```

---

## 3. MATH Hard (Level 5)

MATH Hard is the subset of Level 5 (hardest) problems from the MATH benchmark. These are competition-level math problems from the AMC 10, AMC 12, and AIME, spanning seven subjects: Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, and Precalculus. Each problem has a step-by-step solution with a final boxed answer.

### Stage 1: Capability Annotation

**System Prompt**

```
Given a challenging competition-level mathematical question and its correct solution, generate a gerund phrase that thoroughly and precisely describes the **specific** mathematical skill or capability required to solve the question.
```

**User Prompt**

```
## Question
{input}

## Solution
{output}

## Requirement
- The skill description should be an action-oriented gerund phrase that is **informative** and **detailed**.
- The phrase should refer to a **specific** skill or capability that comprehensively covers the key aspects of the solution, without including any context or specifics from the question or solution.
- Avoid unnecessary elements unrelated to the core capability.
- Please output **only a gerund phrase** describing the skill, with NO additional text.
```

### Stage 4: Capability Description

**System Prompt**

```
Given a set of phrases, each summarizing the mathematical skills or capabilities needed to solve challenging competition-level mathematics questions within a specific group, generate a gerund phrase that summarizes the collective set of mathematical skills or capabilities described across all groups.
```

**User Prompt**

```
## Task
You are given a set of phrases, each summarizing the mathematical skills or capabilities needed to solve challenging competition-level mathematics questions within a specific group. There are {group_number} groups in total. Your task is to **summarize** the collective set of mathematical skills or capabilities that represents the union of these descriptions in a detailed and informative manner.

## Skill Descriptions
{skill_descriptions}

## Requirements
- The output should be a **single gerund phrase** that succinctly summarizes the overarching mathematical skill or capability represented by the union of all the provided phrases.
- The output should comprehensively cover each skill description without going beyond them.
- The output should not simply enumerate the given phrases but instead provide a meaningful and informative summary of the mathematical skills or capabilities they collectively represent.
- Please output **only a gerund phrase** summarizing the mathematical skill or capability, with NO additional text.
```

---

## 4. MMLU-Pro

MMLU-Pro is an enhanced multi-task language understanding benchmark with over 12,000 reasoning-focused multiple-choice questions across 14 domains (Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others). Questions have 10 answer choices and require deeper reasoning than the original MMLU.

### Stage 1: Capability Annotation

**System Prompt**

```
Given a challenging multiple-choice question with ten answer options, testing a model's deep reasoning and expert-level knowledge across diverse domains, generate a gerund phrase that thoroughly and precisely describes the **specific** skill or capability required to determine the correct answer.
```

**User Prompt**

```
## Question
{input}

## Answer
{output}

## Requirement
- The skill description should be an action-oriented gerund phrase that is **informative** and **detailed**.
- The phrase should refer to a **specific** skill or capability that comprehensively covers the key aspects of selecting the correct answer, without including any context or specifics from the question or answer.
- Avoid unnecessary elements unrelated to the core capability.
- Please output **only a gerund phrase** describing the skill, with NO additional text.
```

### Stage 4: Capability Description

**System Prompt**

```
Given a set of phrases, each summarizing the skills or capabilities needed to answer challenging multiple-choice questions testing deep reasoning and expert-level knowledge within a specific group, generate a gerund phrase that summarizes the collective set of skills or capabilities described across all groups.
```

**User Prompt**

```
## Task
You are given a set of phrases, each summarizing the skills or capabilities needed to answer challenging multiple-choice questions testing deep reasoning and expert-level knowledge within a specific group. There are {group_number} groups in total. Your task is to **summarize** the collective set of skills or capabilities that represents the union of these descriptions in a detailed and informative manner.

## Skill Descriptions
{skill_descriptions}

## Requirements
- The output should be a **single gerund phrase** that succinctly summarizes the overarching skill or capability represented by the union of all the provided phrases.
- The output should comprehensively cover each skill description without going beyond them.
- The output should not simply enumerate the given phrases but instead provide a meaningful and informative summary of the skills or capabilities they collectively represent.
- Please output **only a gerund phrase** summarizing the skill or capability, with NO additional text.
```

---

## Notes

- **Capability Embedding** (Stage 2): Consistent across all benchmarks. Prepend the prefix `"The model has the following skill or capability: "` to the annotated capability and feed into a sentence embedding model (e.g., `text-embedding-3-small`).
- **Recursive Clustering** (Stage 3): Also consistent across benchmarks. Use K-Means with cluster numbers from 2 to 10, selecting via Silhouette score.
- **Template variables**: `{input}` = the question/problem text, `{output}` = the correct answer/solution/explanation, `{group_number}` = number of child groups, `{skill_descriptions}` = newline-separated list of children's capability descriptions.
- **MATH Hard vs. MATH**: The prompts for MATH Hard are nearly identical to the paper's original MATH prompts (Table 1, Table 5), with minor wording adjustments ("challenging competition-level") to reflect that only Level 5 problems are included. If you prefer exact consistency with the paper's existing MATH tree, the original prompts from Tables 1 and 5 can be used directly.
- **Determining associated instances**: For assessment purposes (Appendix E.1), you will also need prompts analogous to Tables 22–24. These follow the same benchmark-specific pattern: present the question and answer, then ask whether a given skill or capability is required, expecting a YES/NO output.
