---
layout: post
title: "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs"
date: 2025-11-23
description: Paper reading notes on HippoRAG - using knowledge graphs and PageRank for better RAG retrieval
tags: paper-notes rag llm knowledge-graph pagerank
categories: paper-reading
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

# HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs

This paper propose an new RAG paradigm using knowledge graph, which is inspired by neural science and hippocampus.

# Main goal

The main goal is still to get the most relevant passages (paragraphs) given user query.

Still a traditional RAG.

It is NOT: trying to answer queries directly from RAG

It is NOT: returning entities to the LLM

In short, this paper use PageRank algorithm to find the optimal path/tree for entities and return the chunks containing most of those entities.

# Main architecture

The HippoRAG method takes two main steps: Offline Indexing, and Online Retrieval

## Some notations

$P$ : number of passages

$L$: LLM

$M:$ retrieval Encoder

$N:$ set of noun nodes (entities)

$E:$ relation edges

## Offline Indexing

**For each passage P:**

Step 1: use L to extract N, the prompt is as the follows:

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_13.17.19.png" width="60%" alt="Screenshot 2025-11-18 at 13.17.19.png">

Step 2: and use OpenIE (an NLP LLM) to extract the triplet. This will create relation edges E among all the entities (within this passage). This relation is of type **Fact**.

The prompt used in this step is

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_13.17.58.png" width="60%" alt="Screenshot 2025-11-18 at 13.17.58.png">

**After done step 1-2 for each passages**

Step 3. Now we aggregate all the unique entities and facts across all passages. This will give us a global Knowledge Graph (KG).

Step 4. Additionally, we can use a encoder model M to add synonymy relations if there similarity is above a threshold $\tau$.

One example is:

Given two passage as the following:

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_13.32.32.png" width="60%" alt="Screenshot 2025-11-18 at 13.32.32.png">

Then NER and OpenIE triplet of them are:

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_13.33.24.png" width="60%" alt="Screenshot 2025-11-18 at 13.33.24.png">

And the final graph (of these two passages) is given as:

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_13.34.28.png" width="60%" alt="Screenshot 2025-11-18 at 13.34.28.png">

Note that there are:

- 3-types of edges:
  1. Fact-edges: entity ↔ entity
  2. Passage-edges: passage ↔ entity (connect each passage, to ALL entities that appear in its triplets)
  3. Synonymy-edges: entity ↔ similar entity
- 2-types of nodes
  1. entity-node
  2. passage-node: in this case, passages with shared entities are ‘connected’.

After this process, we have the following matrix

$$
\mathbf{P} \in \mathbb{R}^{|N|\times|P|}
$$

where each row corresponds to each entity and how many times it appears in each passages (columns).

## Online Retrieval

**Step 1.** Given a query $q$, use the same prompt + LLM to extract named entities

$$
C_q = \{c_1, \ldots, c_n \}.
$$

The named entities $C_q$ is then mapped to $N$ (the set of entities from KG) using the max cos_sim, i.e., we got the so-called query nodes defined as

$$
R_q = \{ r_1, \ldots, r_n \}
$$

where $r_i$ is the closest entity to $c_i$ from $N$, namely

$$
r_i = \arg\max_{e_j\in N} cossim (M(c_i), M(e_j))
$$

where M is the same encoder model.

**One risk:** there is a potential risk that some of the entity from the query are not properly matched, or even missed! For example:

```python
query: "What county is Erik Hort's birthplace in?"
C_q = ["Erik Hort", "birthplace", "county"]
```

This entity “country” might be matched pretty randomly. Or just one of the entity, say “Erik Hort” got matched. In either case, we end up with very limited starting state in the graph (later for PPR), or very misleading state.

There is another way of doing this step 1:

**Step 1’.** In the repo of HippoRAG, they actually did not run directly NER on query (step 1), but

retrieve the facts (relations). For example:

```python
query: "What county is Erik Hort's birthplace in?"
```

Then we rerank facts using the same encoder model

```python
Top facts: [("Erik Hort", "birthplace is", "Montebello"),
            ("Montebello", "is a part of", "Rockland County")]
```

Then use the entities from these selected fact relations as seeds in PPR later.

Either step 1 and 1’ is trying to get some init seed in the PPR alg. Especially the ones from Step 1’ is called SEED nodes.

**Step 2.** After get the query nodes $R_q$ (seed nodes), PPR is excuted over the KG ( $|N|$ nodes and $|E|+|E'|$ edges (fact + synonymy).

In this page we review the original pagerank alg.

Here the PPR (personalized PageRank)

```markdown
At each step:

- With probability α (damping): Follow a random outgoing edge
- With probability (1-α): Jump back to SEED nodes (not uniform!)
```

Stationary distribution is then given as

$$
\pi = \alpha \pi M + (1-\alpha) s
$$

where:

- M = transition matrix (based on graph structure)
- s = personalized reset distribution (seed nodes)

The calculations of M and s are discussed below.

**Calculation of M**

So in this step, it is essentially about how we design the transition matrix.

M is defined based on weighted edges.

The weight of edges has 3 sources.

1. Fact edges: Weight = Number of times this entity pair appears in triples

```python
self.node_to_node_stats[(node_key, node_2_key)] += 1
self.node_to_node_stats[(node_2_key, node_key)] += 1

# ("CEO", "company") appears in 5 triples → weight = 5
```

1. Passage Edges: weight = 1.0
2. Synonymy Edges: weight = simiarity score ( between 0 1).

Then given edge weights, the transition prob from node i to node j is given as

$P(i → j) = \frac{weight(i,j)}{ \sum_k weight(i,k)}$

**Calculation of s (reset distribution, reset prob, whatever)**

1. It is not uniform.
2. It is weighted by: Entity nodes and Passage nodes. For example

```python
Query = "What county is Erik Hort's birthplace in?"

#Top-k Facts (relations, step 1')
top_k_facts = [
    ("Erik Hort", "birthplace is", "Montebello"),      # score: 0.9
    ("Montebello", "is a part of", "Rockland County"), # score: 0.8
    ("Oliver Badman", "is a", "politician")            # score: 0.3
]

# Entity weights from facts
phrase_weights[entity_id("Erik Hort")] = 0.9
phrase_weights[entity_id("Montebello")] = (0.9 + 0.8) / 2 = 0.85  # Appears in 2 facts
phrase_weights[entity_id("Rockland County")] = 0.8
phrase_weights[entity_id("Oliver Badman")] = 0.3
phrase_weights[entity_id("politician")] = 0.3

# Passage weights (small, from dense retrieval)
passage_weights[passage_id("P1")] = 0.02
passage_weights[passage_id("P2")] = 0.01

# Reset distribution
reset_prob = phrase_weights + passage_weights
# Normalize: reset_prob = reset_prob / sum(reset_prob)
```

Here the $s$ is not the one from paper section ‘Node Specificity’ which is just a parameter in front of initial distribution $\bar{n}$ (defined below).

Then execution of PPR is just run the iteration until it converges. The initial distribution can be uniform among all query named entities, and other nodes with 0 probability.

Denote the init distribution as $\bar{n}$.

During PPR, the probability mass is distributed to nodes that are primarily in the (joint) neighborhood of the query nodes.

Let us denote the converged probability distribution over $N$ as $\bar{n'}$, **to obtain passage score, we need to**

$$
\bar{p} = \bar{n'} \mathbf{P}
$$

$\mathbf{P}$ is from the offline indexing, and this is a row vector multiply with a matrix, gives a row vec in $\mathbb{R}^{|P|}$.

Then the top passages are returned to LLM.

# Conclusion

This paper provides a systematic way to find the most relevant entities to the query, and then the most relevant document chunks for LLM. I think this could be a potentially good method for structured data/table retrieval. More precisely, when we need to return several tables for a SQL generation, this method could give us an optimal path for table joins, namely 1-hop or 3-hop neighbors.

# Some background info

# PageRank

This the fundamental alg for Google to rank millions webpages. This is based on an assumption that user can start with random webpage and with some probability to perform a random walk. The becomes a Markov Chain.

# markov chain

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_14.56.17.png" width="60%" alt="Screenshot 2025-11-18 at 14.56.17.png">

Then the distribution of being each node is governed by

$$
\pi_{n+1} = \pi_n P
$$

where $P$ is the transition probability matrix.

## Stationary distribution:

$$
\pi = \pi P
$$

Theorem: For irreducible and aperiodic Markov chains:

1. A unique stationary distribution $\pi$ exists
2. All initial distributions $\pi_0$ converge to $\pi$

Here irreducible basically means the graph is connected, and aperiodic mean the graph is not oscillating between two nodes.

### Calculation of stationary distribution

1. Brutal

Just start with any initial distribution, times multiple $P,$ i.e., $\pi_0 P^n$ until the residual smaller than a epsilon. Most practical

1. Using linear equation solver
2. Egenvector

$$
\pi^T = P^T \pi^T
$$

So $\pi^T$ is the eigenvector of $P^T$ of egenvalue 1.

# PageRank

Essentially calculating stationary distribution of a Markov chain.

But with some variation:

1. Webpage with NO outgoing links. All eventually stop at that node

   <img src="/assets/img/hipporag/Screenshot_2025-11-18_at_15.10.27.png" width="60%" alt="Screenshot 2025-11-18 at 15.10.27.png">

2. Cycles: Also end up with state 1 and 2, oscillating forever.

   <img src="/assets/img/hipporag/Screenshot_2025-11-18_at_15.11.18.png" width="60%" alt="Screenshot 2025-11-18 at 15.11.18.png">

PageRank is handling with Markov Chains that are NOT irreducible and aperiodic!

Idea: if the user is stuck in a state, randomly select a new state

Step 1. Given a parameter $\alpha\in(0,1)$, with probability $\frac{\alpha}{N}$ where N is the number of states (nodes), we can transition to a random state.

For example

<img src="/assets/img/hipporag/Screenshot_2025-11-18_at_15.17.15.png" width="60%" alt="Screenshot 2025-11-18 at 15.17.15.png">

But now, the probability sum is NOT 1 anymore.

Step 2. Adding a damping factor: reduce original transition probabilities by factor of $(1-\alpha)$ and then add the random state transition $\frac{\alpha}{N}$.

Mathematically, we modify the matrix $P$ to $\bar{P}$

$$
\bar{P} = (1-\alpha) P + \frac{\alpha}{N} \mathbf{1}_{N\times N}
$$

that is N by N all 1 matrix.

PageRank is then just finding the stationary distribution of $\bar{P}$.
