---
layout: post
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
date: 2026-01-01
description: "Notes on ReAct - a paradigm for LLM agents that combines reasoning traces and task-specific actions. Covers prompt engineering, fine-tuning approaches, and implementation of zero-shot ReAct agents with LangChain."
tags: react llm-agents prompt-engineering reasoning chain-of-thought langchain
categories: paper-reading
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

This is the most cited work by Shunyu Yao so far.

This paper is essentially about prompt engineering, but defines a paradigm of interacting with LLM, especially with explosion of LLM agent. In fact, more recent LLMs have reasoning capability as builtin capability, which essentially means ReAct is enabled by default. ReAct enjoys several unique features:

1. Intuitive and easy to design: it is close to human way of solving problems. In the prompt, just need the human annotator write down their thoughts during problem-solving.
2. Less hallucination comparing to Chain-of-thought.

# Core idea of ReAct

The basic interaction of an agent with an environment for a task is composed by

- at time $t$, agent receives an observation $o_t\in \mathcal{O}$ from the env
- takes an action $a_t\in\mathcal{A}$ following some policy $\pi(a_t|c_t)$
- $c_t = (o_1, a_1, \ldots, o_{t-1}, a_{t-1}, o_t)$ is the context to the agent.

The authors proposed a very reasonable and intuitive expansion: augment the action space as $\hat{\mathcal{A}} = \mathcal{A}\cup \mathcal{L}$ , where $\mathcal{L}$ is the space of language (for thought and reasoning). Now, besides the action and observation, the full interaction between an agent and an env is expanded with ‘thought’ which is

- free-form language thoughts
- decompose goals,
- track progress
- handle exceptions

# Prompt design

This paper proposed some prompt templates for different use case. Note all these cases is using few-shot prompt, namely no special prefix design but including some examples in the prompt.

### HotpotQA

**Dense thoughts**: this eval set is more knowledge intensive, hence a thought accompanies every action to guide the search and synthesis of the info.

One example is attached as following:

<img src="/assets/img/react/Screenshot_2025-12-26_at_10.07.08.png" width="60%" alt="Screenshot 2025-12-26 at 10.07.08.png">

<img src="/assets/img/react/Screenshot_2025-12-26_at_10.07.42.png" width="60%" alt="Screenshot 2025-12-26 at 10.07.42.png">

<img src="/assets/img/react/Screenshot_2025-12-26_at_10.07.58.png" width="60%" alt="Screenshot 2025-12-26 at 10.07.58.png">

this type is for dense thought, for sparse thought, the model dose not need to think at every step, but asynchronously as needed.

# Why ReAct can be learned by LLM?

First, this paper discussed fine-tune a LLM to boost the ReAct performance. The author used a base model PaLM-8/62B as a candidate. Without finetuning, ReAct perform the worst. But with only 3000-example fine-tuning, the ReAct outperform all the others.

<img src="/assets/img/react/Screenshot_2025-12-30_at_10.48.54.png" width="60%" alt="Screenshot 2025-12-30 at 10.48.54.png">

The authors also compared the ReAct Performance of PaLM-540B and GPT3. GPT3 is better means the model there can follow the ReAct prompt better than PaLM-540B. This is probably because the instruct fine tuning done towards GPT3 with some human thinking and action steps in the samples.

<img src="/assets/img/react/Screenshot_2025-12-30_at_13.02.41.png" width="60%" alt="Screenshot 2025-12-30 at 13.02.41.png">

# In-context-learning and zero-shot ReAct agent

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

If we ask the agent "If Wolfgang Amadeus Mozart had lived as long as Ludwig van Beethoven did, what year would Mozart have died?”, we got the following chain:

```markdown
Reasoning: First, I need to find out when Mozart was born and when he died to calculate his age. Then, I need to find out when Beethoven was born and when he died to calculate his age. If Mozart had lived as long as Beethoven, I can add Beethoven's age to Mozart's birth year to find out the year Mozart would have died.
Action: search_wikipedia
Action Input: "Wolfgang Amadeus Mozart"
Observation: Wolfgang Amadeus Mozart (27 January 1756 – 5 December 1791) was a prolific and influential composer of the Classical period. Despite his short life, his rapid pace of composition and proficiency from an early age resulted in more than 800 works representing virtually every Western classical genre of his time. Many of these compositions are acknowledged as pinnacles of the symphonic, concertante, chamber, opera, and choral repertoires. Mozart is widely regarded as one of the greatest composers in
Reasoning: Now I know that Mozart was born in 1756 and died in 1791. I need to calculate his age.
Action: calculate
Action Input: "1791 - 1756"
Observation: 35
Reasoning: Mozart was 35 years old when he died. Now I need to find out when Beethoven was born and when he died to calculate his age.
Action: search_wikipedia
Action Input: "Ludwig van Beethoven"
Observation: Ludwig van Beethoven (baptised 17 December 1770 – 26 March 1827) was a German composer and pianist. One of the most revered figures in the history of Western music, his works rank among the most performed of the classical music repertoire and span the transition from the Classical period to the Romantic era. Beethoven's early period, during which he forged his craft, is typically considered to have lasted until 1802. From 1802 to around 1812, his middle period showed an individual development fr
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
