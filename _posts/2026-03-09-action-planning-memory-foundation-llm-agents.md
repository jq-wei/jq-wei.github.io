---
layout: post
title: "Action, Planning, and Memory: The Foundation of LLM Agents"
date: 2026-03-09
description: "Exploring three foundational frameworks for LLM agents - ReAct (action and reasoning), Tree of Thoughts (planning and search), and Reflexion (memory and learning) - that transform passive language models into capable autonomous agents."
tags: llm-agents react tree-of-thoughts reflexion reasoning planning memory shunyu-yao
categories: paper-reading
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet their underlying mechanism remains surprisingly simple: they generate text token-by-token in a sequence. However, at least for the basic transformer inference, this generation is passive. To transform these models from passive text generators into capable autonomous agents, we must look beyond standard prompting paradigms. This post explores one foundational framework (main contribution by Shunyu Yao) that provide the missing architectural pillars for true agency:

**ReAct** (equipping models with external actions and observations), **Tree of Thoughts** (upgrading internal reasoning to deliberate, "System 2" planning and search), and **Reflexion** (providing dynamic memory to learn from past successes and failures). Together, action, planning, and memory form the fundamental blueprint for the next generation of AI agents.

# ReAct

This is the most cited work by Shunyu Yao so far.

This work defines a paradigm of interacting with LLM, especially with explosion of LLM agent. In fact, more recent LLMs have reasoning capability as builtin capability, which essentially means ReAct is enabled by default. ReAct enjoys several unique features:

1. Intuitive and easy to design: it is close to human way of solving problems. In the prompt, just need the human annotator write down their thoughts during problem-solving.
2. Less hallucination comparing to Chain-of-thought.

## Core idea of ReAct

The basic interaction of an agent with an environment for a task is composed by

- at time $t$, agent receives an observation $o_t\in \mathcal{O}$ from the env
- takes an action $a_t\in\mathcal{A}$ following some policy $\pi(a_t\|c_t)$
- $c_t = (o_1, a_1, \ldots, o_{t-1}, a_{t-1}, o_t)$ is the context to the agent.

The authors proposed a very reasonable and intuitive expansion: augment the action space as $\hat{\mathcal{A}} = \mathcal{A}\cup \mathcal{L}$, where $\mathcal{L}$ is the space of language (for thought and reasoning). Now, besides the action and observation, the full interaction between an agent and an env is expanded with 'thought' which is

- free-form language thoughts
- decompose goals,
- track progress
- handle exceptions

## Prompt design

This paper proposed some prompt templates for different use case. Note all these cases is using few-shot prompt, namely no special prefix design but including some examples in the prompt.

### HotpotQA

**Dense thoughts**: this eval set is more knowledge intensive, hence a thought accompanies every action to guide the search and synthesis of the info.

One example is attached as following:

<img src="/assets/img/llm-agents/Screenshot_2025-12-26_at_10.07.08.png" width="80%" alt="HotpotQA Example 1">

<img src="/assets/img/llm-agents/Screenshot_2025-12-26_at_10.07.42.png" width="80%" alt="HotpotQA Example 2">

<img src="/assets/img/llm-agents/Screenshot_2025-12-26_at_10.07.58.png" width="80%" alt="HotpotQA Example 3">

This type is for dense thought, for sparse thought, the model dose not need to think at every step, but asynchronously as needed.

## Why ReAct can be learned by LLM?

First, this paper discussed fine-tune a LLM to boost the ReAct performance. The author used a base model PaLM-8/62B as a candidate. Without finetuning, ReAct perform the worst. But with only 3000-example fine-tuning, the ReAct outperform all the others.

<img src="/assets/img/llm-agents/Screenshot_2025-12-30_at_10.48.54.png" width="70%" alt="ReAct Finetuning Results">

The authors also compared the ReAct Performance of PaLM-540B and GPT3. GPT3 is better means the model there can follow the ReAct prompt better than PaLM-540B. This is probably because the instruct fine tuning done towards GPT3 with some human thinking and action steps in the samples.

<img src="/assets/img/llm-agents/Screenshot_2025-12-30_at_13.02.41.png" width="70%" alt="Model Comparison">

## In-context-learning and zero-shot ReAct agent

In this paper, the authors are using standard in-context-learning with few-shot examples in the prompt for the LLM to learn ReAct. Since then it has become a foundational paradigm for LLM agents. Now in langchain, it provide a zero-shot ReAct agent (Langchain incorporate agent endpoint to `create_agent` now, and zero-shot ReAct agent `create_react_agent` is moved to `langchain_classic`). In this section, we explore some behaviors of this type of agent.

The skeleton of the code is as follows, where we defined a zero-shot ReAct agent with 3 basic tools.

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor

# Load environment variables from .env
load_dotenv()

# ============================================================================
# Define Tools
# ============================================================================

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about people, places, events, concepts."""
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia(user_agent='ReActTestAgent/1.0', language='en')
    page = wiki.page(query)
    if page.exists():
        return page.summary[:500]
    return f"No Wikipedia page found for '{query}'."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Examples: '2 + 2', '10 * 5', '(100 - 20) / 4'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Only basic math operations allowed"
    return str(eval(expression))

@tool
def get_current_year() -> str:
    """Get the current year."""
    from datetime import datetime
    return str(datetime.now().year)

# ============================================================================
# Create Agent
# ============================================================================

def create_agent():
    """Create a ReAct agent with explicit reasoning traces."""
    tools = [search_wikipedia, calculate, get_current_year]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # ReAct prompt template from the paper
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True
    )

# ============================================================================
# Main
# ============================================================================

def main():
    agent = create_agent()
    
    while True:
        question = input("Question: ").strip()
        if question:
            print()
            result = agent.invoke({"input": question})
            print(f"\nFinal Answer: {result['output']}\n")

if __name__ == "__main__":
    main()
```

If we ask the agent "If Wolfgang Amadeus Mozart had lived as long as Ludwig van Beethoven did, what year would Mozart have died?", we got the following chain:

```markdown
Reasoning: First, I need to find out when Mozart was born and when he died to calculate his age. Then, I need to find out when Beethoven was born and when he died to calculate his age. If Mozart had lived as long as Beethoven, I can add Beethoven's age to Mozart's birth year to find out the year Mozart would have died.

Action: search_wikipedia
Action Input: "Wolfgang Amadeus Mozart"
Observation: Wolfgang Amadeus Mozart (27 January 1756 – 5 December 1791) was a prolific and influential composer of the Classical period. Despite his short life, his rapid pace of composition and proficiency from an early age resulted in more than 800 works representing virtually every Western classical genre of his time...

Reasoning: Now I know that Mozart was born in 1756 and died in 1791. I need to calculate his age.
Action: calculate
Action Input: "1791 - 1756"
Observation: 35

Reasoning: Mozart was 35 years old when he died. Now I need to find out when Beethoven was born and when he died to calculate his age.
Action: search_wikipedia
Action Input: "Ludwig van Beethoven"
Observation: Ludwig van Beethoven (baptised 17 December 1770 – 26 March 1827) was a German composer and pianist...

Reasoning: Now I know that Beethoven was born in 1770 and died in 1827. I need to calculate his age.
Action: calculate
Action Input: "1827 - 1770"
Observation: 57

Reasoning: Beethoven was 57 years old when he died. Now, if Mozart had lived as long as Beethoven, I can add Beethoven's age to Mozart's birth year to find out the year Mozart would have died.
Action: calculate
Action Input: "1756 + 57"
Observation: 1813

Reasoning: I now know the final answer
Final Answer: If Wolfgang Amadeus Mozart had lived as long as Ludwig van Beethoven did, Mozart would have died in the year 1813.
```

# ToT

Tree of Thoughts is a framework that empowers LMs to more autonomously and intelligently make decisions and solve problems. It is motivated by cognitive research back to 1950s.

*A genuine problem-solving process involves the repeated use of available information to initiate exploration, which discloses, in turn, more information until a way to attain the solution is finally discovered.—— Newell et al.*

In this paper, ToT is limited to pure reasoning and search problems that do not require external interaction. But it fits naturally with ReAct agent paradigm. It can be seen as the guide for the thinking process of the agent, and the state can contain the observation after apply the action proposed by the thought.

## Notation

- $p_\theta$: pretrained LM with parameter $\theta$
- lowercase letter to denote a language seq: $x = (x[1], \cdots, x[n])$ where $x[i]$ is a token
- uppercase letter to denote a collection of seq: $S$
- Input-output prompting: $y \sim p_\theta^{IO}(y \| x)$ where x is the total input to the LM.

## Chain of thought

Given the notation above, we can denote the CoT, which is a one-shot LLM call, as

$$
[z_{1...n}, y] \sim p_\theta^{CoT}(z_{1...n}, y | x)
$$

where $z_i$ is the each thought and y is the final output. In other words, in CoT, the LLM samples the entire sequence of intermediate thoughts and the final output as one **"continuous language sequence"** based on the initial prompt.

## Tree of thought

### General framework

<img src="/assets/img/llm-agents/Screenshot_2026-03-08_at_07.31.43.png" width="70%" alt="Tree of Thoughts Framework">

Sequence generations like CoT suffer from these two key shortcomings:

- *Locally, they do not explore different continuations within a thought process – the branches of the tree.*
- *Globally, they do not incorporate any type of planning, lookahead, or backtracking to help evaluate these different options – the kind of heuristic-guided search that seems characteristic of human problem-solving.*

ToT frames any problem solving process as a search over a tree, where each node in the tree is a state $s = [x, z_{1\ldots i}]$ representing a partial solution with the input and the sequence of thoughts so far.

ToT address these shortcomings from four perspectives:

1. *How to decompose the intermediate process into thought steps;*
2. *How to generate potential thoughts from each state;*
3. *How to heuristically evaluate states;*
4. *What search algorithm to use.*

**1 Thought decomposition**

This part is task dependent, but the general guide is: a thought should be "small" enough so that LMs can generate promising and diverse samples, yet "big" enough so that LMs can evaluate its prospect toward problem solving.

**2 Thought generator**

Given state $s = [x, z_{1\ldots i}]$, the authors promote two strategies to generate k candidates for the next thought:

- Sample: $z^{(j)} \sim p_\theta^{CoT}(z_{i+1} \| s) = p_\theta^{CoT}(z_{i+1} \| x, z_{1...i}), j = 1 \cdots k$. This is suitable if the thought space is rich (continuous), for example creative writing.
- Propose: $[z^{(1)}, \cdots, z^{(k)}] \sim p_\theta^{propose}(z_{i+1}^{(1 \cdots k)} \mid s).$ This is more suitable for tasks with a more constrained thought space.

**3 State evaluator**

Now with the multiple thoughts generated, state evaluator evaluates the progress they make towards solving the problem. Here the authors proposed two strategies to evaluate (both using LLM):

- Value: Rate a state with a scalar (0 to 10), or classification (sure/likely/impossible)
- Vote: this is when problem success is hard to directly value, then LLM will vote for the most promising one.

**4 Search algorithm**

<img src="/assets/img/llm-agents/Screenshot_2026-03-09_at_07.08.00.png" width="70%" alt="Search Algorithms">

Given the increasing size of the tree, the authors proposed two algorithms to search for the most promising path while keeping the tree manageable.

BFS explores the problem layer by layer. At each step, it generates candidates, evaluates them, and keeps only the top *b* most promising states to carry forward. Note that even with b=1 (tree becomes a chain), ToT is more promising than CoT, since each kept thought is evaluated.

DFS dives into the single most promising path first. If the state evaluator determines that a current state is "impossible" to solve from (a dead end), DFS actively prunes that subtree and **backtracks** to the parent state to try the next best alternative.

### Example

In the paper, the author consider three experiments, Game24, creative writing, and crosswords. Here we just take the setup of Game24 as an example to enhance our understanding about ToT.

Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24.

<img src="/assets/img/llm-agents/Screenshot_2026-03-09_at_07.17.04.png" width="70%" alt="Game of 24 Example">

For the thought decomposition, this game has at most 3 steps (4 initial numbers, one fewer after each arithmetic operation). Thought generator in this case is at each step, one LLM call to propose some possible moves. Then one separate LLM call to evaluate each thought candidate as 'sure/maybe/impossible' wrt reaching 24. Then BFS is employed to keep the best b=5 candidates at each step.

### Result

<img src="/assets/img/llm-agents/Screenshot_2026-03-09_at_09.59.08.png" width="70%" alt="ToT Results">

For the Game24, the result of ToT is outperforming CoT, even with b=1. From the error analysis (fig 3(b)), CoT failed mostly at the first step of the reasoning, then it has no capability to 'correct itself'. However, ToT fails more 'equally' at each step. This is mainly due its ability to explore and self-correct. The final column in fig.3(b) showcased that ToT has much higher success rate compare to CoT (consistent to the result in Table 2).

# Reflexion

This work is considered as the memory part of the ReAct paradigm. Now with ReAct paradigm and Tree of Thought, the latency for the agent to the solve the task can be huge. Some knowledge saving along the process can be very helpful in this case.

Reflexion is analogy to basic reinforcement learning setting, but with verbal information as policy improver, or memory. Reflexion converts feedback from the environment (binary or scalar) into verbal feedback (as textual summary). Reflexion uses that verbal information to help agents learn from prior failings by adding it as additional context.

Generating these reflective feedback depends on the task. In this paper, the author considered 3 scenarios

- decision-making: AlfWorld, a suite of text-based environments that challenge an agent to solve multi-step tasks
- reasoning: HotpotQA, a Wikipedia-based dataset with QA pairs
- programming: Python and Rust code writing on MBPP, HumanEval, and LeetcodeHardGym

The following example (HotpotQA) illustrate the workflow. The first trial failed, then reflections is generated by a LLM, which is used as context in trial 2.

<img src="/assets/img/llm-agents/Screenshot_2026-03-03_at_10.59.53.png" width="80%" alt="Reflexion Example">

The experiment in this paper shows the power of Reflexion by outperforming agents like ReAct in ALFWorld.

<img src="/assets/img/llm-agents/Screenshot_2026-03-03_at_11.03.23.png" width="70%" alt="Reflexion Performance">

## Architecture

<img src="/assets/img/llm-agents/Screenshot_2026-03-03_at_11.07.18.png" width="70%" alt="Reflexion Architecture">

Reflexion is taking ReAct paradigm and has 4 main components (besides the Env): Actor, Evaluator, Self-reflection, and Memory. All the first three are LLMs, which can be different models, or same model different calls. Memory in this work is text.

**Actor**. This is the same as in ReAct where a LLM is prompted to generate an action. In Figure 2(a), if there is only trajectory as input to Actor, then we fallback to ReAct.

**Evaluator**. This can be a LLM call as well to assessing the quality of the trajectories so far, or at the end of the trail (case by case).

**Self-reflection**. This can be a event-triggered LLM call, for instance a failure signal, and write down textual suggestions/guidance as 'long-term' memory for the future trials. This process of trial-error-reflection can rapidly improve the agent's performance, and very consistent with human behavior.

**Memory**. In this work, there are two types of the memories, short/long-term. The current trajectory of ReAct is short-term and the one from self-reflection is for long. Both are provided as context for the actor.

# My take away

A truly capable autonomous agent requires all three pillars. It uses **ReAct** to interact meaningfully with the world, **ToT** to deliberately plan its choices through search and self-evaluation, and **Reflexion** to learn from its experiences (failure in Reflexion, but perhaps success can help in other tasks) and improve over time. Together, they transform LLMs from passive text generators into active, deliberate, and continuously learning agents.
