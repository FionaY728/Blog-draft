# Integrate Sglang Into Sllm Overview

## Table of Contents

- [ServerlessLLM Architecture Overview](#serverlessllm-architecture-overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [ServerlessLLM Store New CLI](#serverlessllm-store)
  - [ServerlessLLM Store](#serverlessllm-store)
  - [Conclusion and Future Work](#conclusion-and-future-work)


## Introduction of SLLM Store

*Sllm-store* introduces an intelligent multi‑tier storage and rapid checkpoint loading mechanism that maximises storage bandwidth utilisation, ensures stable and predictable loading performance, and maintains framework‑agnostic compatibility, thereby enabling ServerlessLLM to deliver low‑latency inference on demand.We make a set of common assumptions about checkpoints:  


This article will walk you through the system architecture of *sllm* and aims to answer the following questions:

- How will the new design of CLI **start,load,save** improves the usability via **Clicks**?
- How is the current  **sglang working**?
- How do we **save and load model of sglang** checkpoints in sllm version?

The goal of this summer internship is to gain a deep understanding of serverless LLM architectures and implement support for the new SGLang backend, enabling fast model checkpoint loading.

## ServerlessLLM Click New Design
Model management utilities—such as loading and saving models or LoRA adapters—are currently scattered throughout the examples directory (https://github.com/ServerlessLLM/ServerlessLLM/tree/main/sllm_store/examples), making them difficult to find and use. Centralising these commands would improve usability and streamline the developer experience. Click’s decorator-based API, built-in colour styling and interactive prompts, plus one‑line shell completion, make CLI development concise, powerful, and a delight to use.

<p align="center">
  <img src="./storev1.png" alt="storev1.png" width="600">
</p>



- **User Interface**: Includes a CLI for model and cluster management and an API gateway that routes control messages to the controller and inference requests to the appropriate router.
- **SLLM Store -start**: Use sllm-store start with flags like --host, --port, --storage-path, --num-thread, --chunk-size, --mem-pool-size, --disk-size, and --registration-required to launch the checkpoint store gRPC server (e.g. sllm-store start --storage-path $PWD/models --mem-pool-size 4GB).
- **SLLM Store -save**: selecting your HuggingFace model, backend (vLLM/Transformers/sglang), LoRA adapter, tensor parallelism, and local storage paths.
- **SLLM Store -load** Use sllm-store load with options like --model, --backend, --adapter-name, --precision, --tensor-parallel-size and --storage-path to fetch and configure a model from the local checkpoint store.

Just remember: before saving or loading a vLLM/sglang backend model with sllm-store, you need to apply the required patch to keep everything running smoothly.From here, we’ll dive into our new sglang backend that equips sllm with seamless, high‑performance support.

![image.png](./images/outlines1.png)

**Step 1: Deploy an LLM**

![arch_step1.jpg](./images/arch_step1.jpg)

Starting with the control plane, model developers use *sllm* to deploy a model, specifying its name on the HuggingFace hub. Upon receiving the model registration request, the controller processes configurations such as backend choice (e.g., [Transformers](https://github.com/huggingface/transformers) or [vLLM](https://github.com/vllm-project/vllm)), auto-scaling settings (e.g., concurrency limits, minimum/maximum instances), and resource requirements (e.g., CPU and GPU allocation).

Once configured, the controller creates a router for the model and registers its checkpoint with the store manager. The store manager determines an initial server for downloading model checkpoints, then instructs *sllm-store* to download and convert the model into the optimized format. This approach ensures the model is ready for rapid loading and minimal latency during cold starts.

**Step 2: Query the LLM (Simple Case, No Cold Start)**

![arch_step2.jpg](./images/arch_step2.jpg)

In the data plane, when a model inference request is received, the API gateway routes it to the corresponding model router. The router selects one of the available backend instances and forwards the request. Each backend instance operates on one or more dedicated GPUs within a shared server, processing the inference request with an inference engine (such as *Transformers* or *vLLM*) and returning the result to the client.

**Step 3: Query the Model with Cold Start**

![arch_step3.jpg](./images/arch_step3.jpg)

In a cold-start scenario, when traffic exceeds capacity, the router scales up by creating additional backend instances. It issues a resource allocation request to the storage-aware scheduler, which selects an optimal server based on model details (e.g., size), hardware specifications (e.g., PCIe and disk bandwidth), and storage status (e.g., whether the model is stored on the server’s local disks or host memory). The request router then initiates a new inference backend, which loads the model from the checkpoint store before becoming available.

In summary, these steps illustrate how ServerlessLLM efficiently deploys, serves, and scales LLMs, achieving low-latency serverless inference even with frequent cold starts. The figure below illustrates the interplay between these components.

![arch_full.jpg](./images/arch_full.jpg)

## ServerlessLLM Store

<p align="center">
  <img src="./images/sllm-store.jpg" alt="sllm-store.jpg" width="650">
</p>

ServerlessLLM Store enables fast checkpoint loading with two core modules:

- A checkpoint parser that saves and restores model checkpoints in a cold-start optimized format (detailed in Step 1 below).
- A dedicated checkpoint manager on each GPU server that loads checkpoints into GPUs efficiently and caches frequently used ones in host memory.

Built on these core modules, ServerlessLLM Store offers a two-level Python API:

- A lower-level tensor API that saves and restore tensors for each specific deep learning library. For examples, PyTorch API for saving and loading a PyTorch `state_dict` .
- A higher-level model API, built on the tensor API, that saves and loads models for inference libraries like *Transformers* and *vLLM*.

To illustrate, let’s walk through two steps: 1) saving a *Transformers* pre-trained model into the *sllm* cold-start optimized format, and 2) loading the *sllm* checkpoint to restore a *Transformers* pre-trained model.

<p align="center">
  <img src="./images/outlines2.png" alt="outlines2.png" width="700">
</p>

**Step 1: Save a Model**

<p align="center">
  <img src="./images/save_model.jpg" alt="save_model.jpg" width="400">
</p>

The`save_model`function takes a *Transformers* pre-trained model and an output path as inputs. It first saves model configurations using *Transformers*’ built-in API, then calls the PyTorch API (`sllm_store.torch.save_dict`) to save the model’s `state_dict` in a cold-start optimized format.

The `save_dict` function uses the checkpoint parser via `save_tensors`, which saves each tensor’s data in a binary file and returns its offset within that file. After saving tensors, the `save_dict` function records tensor metadata and offsets to an index file. This setup enables efficient retrieval during model loading.

**Step 2: Load a Model**

<p align="center">
  <img src="./images/load_model.jpg" alt="load_model.jpg" width="500">
</p>


The `load_model`function takes a model path as input and returns a *Transformers* pre-trained model. It initializes the model with saved configurations and concurrently calls the PyTorch API (`sllm_store.torch.load_dict`) to load the tensors.

The PyTorch API allocates GPU memory for each tensor, calling the standalone checkpoint manager (via gRPC) to load tensor data into designated addresses. Simultaneously, `load_model` uses the checkpoint parser to restore tensors based on the saved tensor metadata, inferring actual GPU memory addresses using the base GPU memory address and saved tensor offsets.

Before returning the model, a final sync call is sent to the checkpoint manager to ensure all data has loaded correctly.

## Conclusion and Future Work

In the next blog post, we’ll demonstrate a deployment example of Serverless RAG. Future posts will also explore specific topics in greater detail, including the scheduling algorithm, cold-start optimized checkpoint format, and the efficient multi-tier checkpoint loading pipeline.
