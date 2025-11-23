---
layout: post
title: Main topics I worked on in HKRC Huawei
date: 2025-07-15
inline: false
related_posts: false
---

#### Agentic RAG acceleration

<ul>
    <li>For the central control unit of XiaoYi AI Agent system, we accelerate the performance (TTFT) to more than 4x, by using more efficient attention mechanism</li>
    <li>SFT (32B model) pipeline which guarantees the accuracy loss < 1%</li>
</ul>

#### Hardware-based inference optimization

<ul>
    <li>Optimize the decoding attention kernel (a key kernel of the inference framework) based on the GQA and NPU architecture</li>
    <li>This solution achieved 4x speed-up and was delivered as Ascend NPU Kernel (decoding attention) for xiaoyi commercial vLLM inference framework</li>
</ul>
