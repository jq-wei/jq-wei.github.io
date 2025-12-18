---
layout: post
title: "A Survey on In-Context Learning"
date: 2025-12-18
description: "Survey notes on in-context learning (ICL) - a paradigm for LLMs to learn tasks from few examples without parameter updates. Covers ICL mechanisms, prompt engineering, and comparison of causalLM vs prefixLM."
tags: in-context-learning icl llm prompt-engineering few-shot-learning
categories: paper-reading
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

This paper is from Peking U, ByteDance, Shanghai AI Lab, Alibaba.

Nowadays in-context-learning become a good practice when we use LLM to solve real-world task with some pattern or style. This paper is a good entry point to understand the research landscape of ICL (in context learning).

In this note I record my learning note on this domain, with deep dive into some papers in the reference.

The key idea of in-context learning is to learn from analogy (类比) usually with a few examples in the prompt.

ICL does not perform parameter updates. The model is expected to learn the pattern hidden in the demonstration and accordingly make the right prediction

# Advantages of ICL

1. It is much easier to incorporate human knowledge into LLMs by changing the demonstration and templates.
2. ICL is similar to the decision process of human beings by learning from analogy
3. ICL is training free.

# Definition

In-context learning is a paradigm that allows language models to learn tasks given only a few examples in the form of demonstration.

Formally, given a query $x$ and a set of candidate answers $Y=\{y_1, \ldots, y_m\}$ (here $Y$ can be class labels, or a set of free-text phrases), a LLM $\mathcal{M}$ takes the candidate answer with the max score as the prediction, conditioned a demonstration set $C$.

Here $C$ can be $\{I, s(x_1, y_1), \ldots, s(x_k, y_k)\}$ where $s(x_i, y_i)$ is an example pair, $I$ is an instruction (optional), and these $y_i$ in $C$ are different from $Y$.

Here depending on whether examples in $C$ belong to the same task or not, ICL can be classified as task-specific ICL and cross-task ICL

### Some terminology clarification

Few shot learning: a general ML approach that involves adapting model parameters to perform a task with a limited number of supervised examples.

ICL does not train the model, hence no weights are modified.

# Arena of ICL

## How did the ICL capability happen

1. pretraining or continual pretraining
   1. aggregating related contexts, making the models learn to reason across prior demonstrations.
2. Warmup
   1. this is a way to enhance ICL ability
   2. adding a continual training stage between pretraining and ICL inference.
   3. this modify or add LLM parameters.
   4. most of the works are in 2022-2023.

## Prompt designing

Major usage of ICL is during inference. This section is about how to boost the performance of ICL by prompt-engineering.

### Demonstration organization

1. Selection: which are the good samples?
   1. One un-surpervised option is: choose the nearest neighbors of input instances based on their similarities
   2. supervised way: train a dense retriever for demonstration selection (use human-labeled or LLM-labeled demonstrations)
   3. RL way: demonstration selection can be a Markov decision process; the action is choosing example; reward is defined as the accuracy of a labeled validation set; selected demonstration can be learned via Q-learning.
2. Formatting: here main stream is to use LLM to reformat the demonstrations
3. Ordering
   1. arrange examples based on their proximity to the input.
   2. use global and local entropy metrics, finding a positive correlation between these metrics and the ICL performance
   3. or gradually increase the complexity of the demonstrations (simple → difficult)

# Why ICL works?

## Influencing factors

1. Pretraining stage

   1. combining multiple source domain is more important to lead to the emergence of ICL (than corpus size)
   2. task diversity
   3. training data with certain distributional properties (bustiness, item appear in clusters, rather than being uniformly distributed over time)
   4. This is an interesting found: during the inference, ICL samples should attend to each other (current causal LLM may lead to suboptimal ICL performance)

1. Inference Stage

   There are some tricks here to order the difficulty of sample, mixture positive/negative pairs, etc can affect the performance of ICL.

## Learning mechanisms of ICL

Here the author list some of the research studying the reason of ICL capability for LLM. From the transformer architecture perspective, it can identify that some specific attention head (”induction head”) can replicate previous patterns for next-token gen. By studying the info-flow in transformer, ICT shows that some label words serve as (attention) anchors, which aggregate and distribute key info for the final pred.

There are also some work focus on bayesian and gradient descent methods.

# Appendix

### CAUSALLM IS NOT OPTIMAL FOR IN-CONTEXT LEARNING

This paper is from Google Research.

This paper provided a simplified version of proof for the empirical evidence that ICL performs better with prefixLM than causalLM.

**Two attention formulas**

Here we clarify the two attention formulas, namely prefixLM and causalLM. We start with general form of attention layer

$$

\begin{aligned}
z_j &\leftarrow z_j + \mathbf{PVZ} \mathtt{softmax} (\mathbf{Z}^T \mathbf{K}^T \mathbf{Q} z_j) \\
    &= z_j + \mathbf{PV} \sum_{i=1}^n z_i\mathtt{softmax}_i (\mathbf{Z}^T \mathbf{K}^T \mathbf{Q} z_j)
\end{aligned}
$$

where $\mathbf{Z} = (z_1, \ldots, z_n)$ is the (embedded) input tokens, and $\mathbf{P}, \mathbf{V}, \mathbf{K},  \mathbf{Q}$ corresponds to output projection, value, key, query projections, respectively.

Attention in this format is called full attention, namely each input token (embedded) $z_j$ can attend to all positions $i\in\{1, \ldots, n\}.$ This type of attention is mainly used in encoder-based models, like BERT, embedding models etc.

For decoder-only model, it is common to use auto-regressive or causal attention, namely

$$
z_j \leftarrow z_j + \mathbf{PV} \sum_{i=1}^j z_i\mathtt{softmax}_i (\mathbf{Z}_{[1,\ldots,j]}^T \mathbf{K}^T \mathbf{Q} z_j)
$$

In this format, each token $z_j$ is restricted to attend only to previous token and itself.

Another version of attention is based on the observation that some tasks can benefit from a prefix sequence such as context or prompt. In this case, the input sequence $\mathbf{Z} = (z_1, \ldots, z_{n'}, z_{n'+1}, \ldots, z_{n})$ where $[1, \ldots, n']$ are prefix tokens, and the remaining are sample (in ICL sense). Then prefixLM suggests the following attention formula

$$
z_j \leftarrow z_j + \mathbf{PV} \sum_{i=1}^{\max (j, n')} z_i\mathtt{softmax}_i (\mathbf{Z}_{[1,\ldots,\max (j, n')]}^T \mathbf{K}^T \mathbf{Q} z_j)
$$

The main conclusion of this paper is that the linearized version of prefixLM is optimal comparing to (linearized) causalLM. But the proof is only under some quite strong assumption (like LSA in the shape of (5) ), so even though the prefixLM sounds more reasonable in ICL, it is really hard to prove in the term of general attention.

Furthermore, the modern CoT makes the causalLM more powerful, hence harder to prove. But this method holds as a valid research direction, especially with more recent LLM like Qwen3 (SFT a ICT version of it with a mask as in prefixLM).
