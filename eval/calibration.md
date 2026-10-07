# Judge calibration: label these answers

For each answer: is **every** factual claim supported by the excerpts above it? Write `yes` or `no` in the `human_faithful` column of `eval/calibration.csv`.

---

## q050

**Question:** how often does EPLB rebalance experts by default?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: serving/expert_parallel_deployment.md | section: Expert Parallel Deployment > EPLB Parameters
Configure EPLB with the `--eplb-config` argument, which accepts a JSON string. The available keys and their descriptions are:

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `window_size` | Number of engine steps to track for rebalancing decisions | 1000 |
| `step_interval` | Frequency of rebalancing (every N engine steps) | 3000 |
| `log_balancedness` | Log balancedness metrics (avg tokens per expert ÷ max tokens per expert) | `false` |
| `num_redundant_experts` | Additional global experts per EP rank beyond equal distribution | `0` |
| `use_async` | Use non-blocking EPLB for reduced latency overhead | `true` |
| `policy` | The policy type for expert parallel load balancing | `"default"` |
| `communicator` | Backend for expert weight transfers: `"torch_nccl"`, `"torch_gloo"`, `"pynccl"`, `"nixl"`,  or `null` (auto) | `null` |

For example:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

Tip: Prefer individual arguments instead of JSON?

    ```bash
    vllm serve Qwen/Qwen3-30B-A3B \
            --enable-eplb \
            --eplb-config.window_size 1000 \
            --eplb-config.step_interval 3000 \
            --eplb-config.num_redundant_experts 2 \
            --eplb-config.log_balancedness true
    ```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: serving/expert_parallel_deployment.md | section: Expert Parallel Deployment > Expert Parallel Load Balancer (EPLB)
While MoE models are typically trained so that each expert receives a similar number of tokens, in practice the distribution of tokens across experts can be highly skewed. vLLM provides an Expert Parallel Load Balancer (EPLB) to redistribute expert mappings across EP ranks, evening the load across experts.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: serving/expert_parallel_deployment.md | section: Expert Parallel Deployment > Configuration
Enable EPLB with the `--enable-eplb` flag.

When enabled, vLLM collects load statistics with every forward pass and periodically rebalances expert distribution.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: serving/expert_parallel_deployment.md | section: Expert Parallel Deployment > Memory Footprint Overhead
EPLB uses redundant experts that need to fit in GPU memory. This means that EPLB may not be a good fit for memory constrained environments or when KV cache space is at a premium.

This overhead equals `NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`.
For DeepSeekV3, this is approximately `2.4 GB` for one redundant expert per EP rank.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: serving/expert_parallel_deployment.md | section: Expert Parallel Deployment > Example Command
Single node deployment with EPLB enabled:

```bash
# Single node with EPLB load balancing
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # Tensor parallelism
    --data-parallel-size 8 \         # Data parallelism
    --enable-expert-parallel \       # Enable EP
    --enable-eplb \                  # Enable load balancer
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

For multi-node deployment, add these EPLB flags to each node's command. We recommend setting `--eplb-config '{"num_redundant_experts":32}'` to 32 in large scale use cases so the most popular experts are always available.
[END DOCUMENT 5]
```
</details>

**Answer:**

EPLB rebalances experts every `3000` engine steps by default, as specified by the `step_interval` parameter [1].

---

## q005

**Question:** if I turn off v1 multiprocessing so pdb works, does that mess with my script's random state?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: design/multiprocessing.md | section: Python Multiprocessing > Prior State in v1
There was an environment variable to control whether multiprocessing is used in
the v1 engine core, `VLLM_ENABLE_V1_MULTIPROCESSING`. This defaulted to off.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

When it was enabled, the v1 `LLMEngine` would create a new process to run the
engine core.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

It was off by default for all the reasons mentioned above - compatibility with
dependencies and code using vLLM as a library.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: usage/reproducibility.md | section: Reproducibility > Default Behavior
In V1, the `seed` parameter defaults to `0` which sets the random state for each worker, so the results will remain consistent for each vLLM run even if `temperature > 0`.

It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs
for workflows such as speculative decoding. For more information, see: <https://github.com/vllm-project/vllm/pull/17929>

Note

    The random state in user code (i.e. the code that constructs [LLM][vllm.LLM] class) is updated by vLLM 
    only if the workers are run in the same process as user code, i.e.: `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

    By default, `VLLM_ENABLE_V1_MULTIPROCESSING=1` so you can use vLLM without having to worry about
    accidentally making deterministic subsequent operations that rely on random state.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: usage/troubleshooting.md | section: Troubleshooting > Breakpoints
Setting normal `pdb` breakpoints may not work in vLLM's codebase if they are executed in a subprocess. You will experience something like:

``` text
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 100, in trace_dispatch
    return self.dispatch_line(frame)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 125, in dispatch_line
    if self.quitting: raise BdbQuit
                      ^^^^^^^^^^^^^
bdb.BdbQuit
```

One solution is using [forked-pdb](https://github.com/Lightning-AI/forked-pdb). Install with `pip install fpdb` and set a breakpoint with something like:

``` python
__import__('fpdb').ForkedPdb().set_trace()
```

Another option is to disable multiprocessing entirely, with the `VLLM_ENABLE_V1_MULTIPROCESSING` environment variable.
This keeps the scheduler in the same process, so you can use stock `pdb` breakpoints:

``` python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: design/multiprocessing.md | section: Python Multiprocessing > Changes Made in v1
There is not an easy solution with Python's `multiprocessing` that will work
everywhere. As a first step, we can get v1 into a state where it does
"best effort" choice of multiprocessing method to maximize compatibility.

- Default to `fork`.
- Use `spawn` when we know we control the main process (`vllm` was executed).
- If we detect `cuda` was previously initialized, force `spawn` and emit a
  warning. We know `fork` will break, so this is the best we can do.

The case that is known to still break in this scenario is code using vLLM as a
library that initializes `cuda` before calling vLLM. The warning we emit should
instruct users to either add a `__main__` guard or to disable multiprocessing.

If that known-failure case occurs, the user will see two messages that explain
what is happening. First, a log message from vLLM:

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

Second, Python itself will raise an exception with a nice explanation:

```console
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: design/multiprocessing.md | section: Python Multiprocessing > Changes Made in v1
the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```
[END DOCUMENT 5]
```
</details>

**Answer:**

Yes, turning off v1 multiprocessing by setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` will allow the random state in user code to be updated by vLLM. This means that the results may not remain consistent for each vLLM run, as the `seed` parameter defaults to `0` when multiprocessing is enabled, ensuring consistent results across runs even if `temperature > 0` [2]. If you disable multiprocessing, you may inadvertently make deterministic subsequent operations that rely on random state [2].

---

## q034

**Question:** what's the default seed in vllm v1?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: usage/reproducibility.md | section: Reproducibility > Default Behavior
In V1, the `seed` parameter defaults to `0` which sets the random state for each worker, so the results will remain consistent for each vLLM run even if `temperature > 0`.

It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs
for workflows such as speculative decoding. For more information, see: <https://github.com/vllm-project/vllm/pull/17929>

Note

    The random state in user code (i.e. the code that constructs [LLM][vllm.LLM] class) is updated by vLLM 
    only if the workers are run in the same process as user code, i.e.: `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

    By default, `VLLM_ENABLE_V1_MULTIPROCESSING=1` so you can use vLLM without having to worry about
    accidentally making deterministic subsequent operations that rely on random state.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: usage/reproducibility.md | section: Reproducibility > Setting the global seed
The `seed` parameter in vLLM is used to control the random states for various random number generators.

If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set accordingly.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: usage/v1_guide.md | section: vLLM V1
**zero configs** by enabling features/optimizations by default.

We see significant performance improvements from upgrading to V1 core engine, in
particular for long context scenarios. Please see performance benchmark (To be
added).

For more details, check out the vLLM V1 blog post [vLLM V1: A Major
Upgrade to vLLM’s Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (published Jan 27, 2025).

This living user guide outlines a few known **important changes and limitations** introduced by vLLM V1. The team has been working actively to bring V1 as the default engine, therefore this guide will be updated constantly as more features get supported on vLLM V1.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: deployment/frameworks/streamlit.md | section: Streamlit > Prerequisites
Set up the vLLM environment by installing all required packages:

```bash
pip install vllm streamlit openai
```
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: README.md | section: Welcome to vLLM
![](./assets/logos/vllm-logo-text-light.png){ align="center" alt="vLLM Light" class="logo-light" width="60%" }
  ![](./assets/logos/vllm-logo-text-dark.png){ align="center" alt="vLLM Dark" class="logo-dark" width="60%" }

<strong>Easy, fast, and cheap LLM serving for everyone
</strong>

<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

Where to get started with vLLM depends on the type of user. If you are looking to:

- Run open-source models on vLLM, we recommend starting with the [Quickstart Guide](./getting_started/quickstart.md)
- Build applications with vLLM, we recommend starting with the [User Guide](./usage/README.md)
- Build vLLM, we recommend starting with [Developer Guide](./contributing/README.md)
[END DOCUMENT 5]
```
</details>

**Answer:**

The default `seed` parameter in vLLM V1 is `0`, which sets the random state for each worker, ensuring consistent results for each vLLM run even if `temperature > 0` [1].

---

## q019

**Question:** whisper endpoint rejects my big audio files, is there a setting to raise the upload size limit?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: serving/online_serving/speech_to_text.md | section: Speech to Text APIs > API Enforced Limits
Set the maximum audio file size (in MB) that VLLM will accept, via the
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable. Default is 25 MB.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: usage/security.md | section: Security > 5. **Restrict Media Download and Decode Sizes:**
| Environment Variable | Default | Description |
| --- | --- | --- |
| `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB` | `256` | Maximum size in MB for a single remote media response. Oversized responses are rejected while streaming before the full body is materialized in memory. |
| `VLLM_MAX_IMAGE_PIXELS` | `178956970` (~179M pixels) | Maximum decoded image size in pixels. Images exceeding this are rejected before raster memory is allocated. Default matches PIL's built-in 2x decompression-bomb threshold (~680 MB for RGB). |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | Maximum compressed filesize in MB for a single audio file. Enforced on all audio inputs (multimodal chat URLs, speech-to-text uploads, data: URLs, and local file paths) before decoding begins. |
| `VLLM_MAX_AUDIO_DECODE_DURATION_S` | `600` | Maximum decoded audio duration in seconds. Prevents compressed audio from expanding into gigabytes of float32 PCM. |
| `VLLM_MAX_AUDIO_DECODE_BYTES` | `268435456` (256 MiB) | Maximum float32 PCM bytes that audio decoding may allocate. Guards against sample-rate forgery where an inflated header sample rate bypasses the duration guard while the actual frame count causes a multi-GiB allocation. |
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: usage/security.md | section: Security > 5. **Restrict Media Download and Decode Sizes:**
Setting any of these to `0` disables the corresponding limit. This is **not
recommended** for deployments exposed to untrusted users, as it removes the
protection against resource-exhaustion attacks.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: usage/security.md | section: Security > 5. **Restrict Media Download and Decode Sizes:**
(256 MiB) | Maximum float32 PCM bytes that audio decoding may allocate. Guards against sample-rate forgery where an inflated header sample rate bypasses the duration guard while the actual frame count causes a multi-GiB allocation. |
| `VLLM_MAX_EMBED_DECODE_BYTES` | `2147483648` (2 GiB) | Maximum bytes a client-supplied embedding payload (`prompt_embeds`, `image_embeds`, `audio_embeds`, `video_embeds`) may allocate once densified. A sparse tensor carries its own declared shape, so a payload of a few hundred bytes can expand into hundreds of GiB. Checked before `to_dense()`, so the memory is never allocated. Set to `0` to disable. |
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: usage/security.md | section: Security > 5. **Restrict Media Download and Decode Sizes:**
Remote media responses and compressed media files can expand into gigabytes of
memory. vLLM enforces download and decode-size limits to prevent out-of-memory
denial of service:
[END DOCUMENT 5]
```
</details>

**Answer:**

You can set the maximum audio file size that vLLM will accept by adjusting the `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable. The default limit is 25 MB, and you can increase this value to allow larger audio files. However, setting it to `0` disables the limit, which is not recommended for deployments exposed to untrusted users as it removes protection against resource-exhaustion attacks [1][2][3].

---

## q043

**Question:** can vllm bench serve gradually increase the request rate during a run?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: benchmarking/cli.md | section: Benchmark CLI > Load Pattern Configuration
vLLM's benchmark serving script provides sophisticated load pattern simulation capabilities through three key parameters that control request generation and concurrency behavior:

##### Load Pattern Control Parameters

- `--request-rate`: Controls the target request generation rate (requests per second). Set to `inf` for maximum throughput testing or finite values for controlled load simulation.
- `--burstiness`: Controls traffic variability using a Gamma distribution (range: > 0). Lower values create bursty traffic, higher values create uniform traffic.
- `--max-concurrency`: Limits concurrent outstanding requests. If this argument is not provided, concurrency is unlimited. Set a value to simulate backpressure. When set, include `client_queue_time` in `--percentile-metrics` to report time spent waiting for the benchmark client's concurrency limit. With a finite `--request-rate`, `e2el_including_client_queue` reports schedule-relative end-to-end latency; it is omitted for `--request-rate=inf`, where all requests arrive at benchmark start.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: benchmarking/sweeps.md | section: Parameter Sweeps > Basic
`vllm bench sweep serve` starts `vllm serve` and iteratively runs `vllm bench serve` for each server configuration.

Tip
    If you only need to run benchmarks for a single server configuration, consider using [GuideLLM](https://github.com/vllm-project/guidellm), an established performance benchmarking framework with live progress updates and automatic report generation. It is also more flexible than `vllm bench serve` in terms of dataset loading, request formatting, and workload patterns.

Follow these steps to run the script:

1. Construct the base command to `vllm serve`, and pass it to the `--serve-cmd` option.
2. Construct the base command to `vllm bench serve`, and pass it to the `--bench-cmd` option.
3. (Optional) If you would like to vary the settings of `vllm serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--serve-params`.

    - Example: Tuning `--max-num-seqs` and `--max-num-batched-tokens`:

    ```json
    [
        {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: benchmarking/sweeps.md | section: Parameter Sweeps > Workload Explorer
`vllm bench sweep serve_workload` is a variant of `vllm bench sweep serve` that explores different workload levels in order to find the tradeoff between latency and throughput. The results can also be [visualized](#visualization) to determine the feasible SLAs.

The workload can be expressed in terms of request rate or concurrency (choose using `--workload-var`).

Example command:

```bash
vllm bench sweep serve_workload \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100' \
    --workload-var max_concurrency \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --num-runs 1 \
    --output-dir benchmarks/results \
    --experiment-name demo
```

The algorithm for exploring different workload levels can be summarized as follows:

1. Run the benchmark by sending requests one at a time (serial inference, lowest workload). This results in the lowest possible latency and throughput.
2. Run the benchmark by sending all requests at once (batch inference, highest workload). This results in the highest possible latency and throughput.
3. Estimate the value of `workload_var` corresponding to Step 2.
4. Run the benchmark over intermediate values of `workload_var` uniformly using the remaining iterations.

You can override the number of iterations in the algorithm by setting `--workload-iters`.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: benchmarking/cli.md | section: Benchmark CLI > Running With Ramp-Up Request Rate
The benchmark tool also supports ramping up the request rate over the
duration of the benchmark run. This can be useful for stress testing the
server or finding the maximum throughput that it can handle, given some latency budget.

Two ramp-up strategies are supported:

- `linear`: Increases the request rate linearly from a start value to an end value.
- `exponential`: Increases the request rate exponentially.

The following arguments can be used to control the ramp-up:

- `--ramp-up-strategy`: The ramp-up strategy to use (`linear` or `exponential`).
- `--ramp-up-start-rps`: The request rate at the beginning of the benchmark.
- `--ramp-up-end-rps`: The request rate at the end of the benchmark.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: cli/README.md | section: vLLM CLI Guide > bench
Run benchmark tests for latency online serving throughput and offline inference throughput.

To use benchmark commands, please install with extra dependencies using `pip install vllm[bench]`.

Available Commands:

```bash
vllm bench {latency, serve, throughput}
```
[END DOCUMENT 5]
```
</details>

**Answer:**

Yes, `vllm bench serve` can gradually increase the request rate during a run by using the ramp-up feature. This feature supports two ramp-up strategies: `linear` and `exponential` [4]. You can control the ramp-up with the following arguments: `--ramp-up-strategy`, `--ramp-up-start-rps`, and `--ramp-up-end-rps` [4].

---

## q032

**Question:** how do i stop vllm from sending anonymous usage stats?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: usage/usage_stats.md | section: Usage Stats Collection
vLLM collects anonymous usage data by default to help the engineering team better understand which hardware and model configurations are widely used. This data allows them to prioritize their efforts on the most common workloads. The collected data is transparent, does not contain any sensitive information.

A subset of the data, after cleaning and aggregation, will be publicly released for the community's benefit. For example, you can see the 2024 usage report [here](https://2024.vllm.ai).
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: usage/usage_stats.md | section: Usage Stats Collection > Opting out
You can opt out of usage stats collection by setting the `VLLM_NO_USAGE_STATS` or `DO_NOT_TRACK` environment variable, or by creating a `~/.config/vllm/do_not_track` file:

```bash
# Any of the following methods can disable usage stats collection
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: usage/security.md | section: Security > Recommendations
- **Public-facing deployments:** Consider setting `VLLM_MAX_N_SEQUENCES` to a value appropriate for your workload (e.g., `64` or `128`) to limit the blast radius of a single request.
- **Reverse proxy layer:** In addition to vLLM's built-in limit, consider enforcing request body validation and rate limiting at your reverse proxy to further constrain abusive payloads.
- **Monitoring:** Monitor per-request resource consumption to detect anomalous patterns that may indicate abuse.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: usage/usage_stats.md | section: Usage Stats Collection > What data is collected?
The list of data collected by the latest version of vLLM can be found here: [vllm/usage/usage_lib.py](../../vllm/usage/usage_lib.py)

Here is an example as of v0.4.0:

Console: Output

    ```json
    {
      "uuid": "fbe880e9-084d-4cab-a395-8984c50f1109",
      "provider": "GCP",
      "num_cpu": 24,
      "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
      "cpu_family_model_stepping": "6,85,7",
      "total_memory": 101261135872,
      "architecture": "x86_64",
      "platform": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
      "gpu_count": 2,
      "gpu_type": "NVIDIA L4",
      "gpu_memory_per_device": 23580639232,
      "model_architecture": "OPTForCausalLM",
      "vllm_version": "0.3.2+cu123",
      "context": "LLM_CLASS",
      "log_time": 1711663373492490000,
      "source": "production",
      "dtype": "torch.float16",
      "tensor_parallel_size": 1,
      "block_size": 16,
      "gpu_memory_utilization": 0.9,
      "quantization": null,
      "kv_cache_dtype": "auto",
      "enable_lora": false,
      "enable_prefix_caching": false,
      "enforce_eager": false,
      "disable_custom_all_reduce": true
    }
    ```

You can preview the collected data by running the following command:

```bash
tail ~/.config/vllm/usage_stats.json
```
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: design/metrics.md | section: Metrics > Unimplemented - `vllm:tokens_total`
Added by <https://github.com/vllm-project/vllm/pull/4464>, but apparently never implemented. This can just be
removed.
[END DOCUMENT 5]
```
</details>

**Answer:**

You can opt out of usage stats collection by setting the `VLLM_NO_USAGE_STATS` or `DO_NOT_TRACK` environment variable, or by creating a `~/.config/vllm/do_not_track` file. Here are the methods you can use:

```bash
# Any of the following methods can disable usage stats collection
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
``` 

These actions will prevent vLLM from collecting and sending anonymous usage data [2].

---

## q008

**Question:** what UID does the non-root vllm user have in the official docker image?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: deployment/docker.md | section: Using Docker > Run as a non-root user
The CUDA `vllm/vllm-openai` image runs as root by default for backward
compatibility. It is also prepared to run as the built-in `vllm` user
(UID 2000, GID 0):

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

When mounting model or cache volumes for a non-root container, mount writable
paths under `/home/vllm` instead of `/root`. For example, mount the Hugging
Face cache at `/home/vllm/.cache/huggingface` and make the mounted directory
writable by group 0.

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

To build an image that defaults to the non-root `vllm` user, use the opt-in
`vllm-openai-nonroot` target:

```bash
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

docker run --rm --gpus all \
    -p 8000:8000 \
    vllm-openai-nonroot:local \
    meta-llama/Llama-3.1-8B-Instruct
```

The `vllm-openai-nonroot` target also supports OpenShift-style arbitrary UIDs
when the runtime UID is a member of group 0. In Kubernetes manifests, set the
container security context accordingly and keep mounted cache/model paths
writable by group 0:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000540000
  runAsGroup: 0
  fsGroup: 0
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: features/initialized_snapshots.md | section: Initialized engine snapshots > Requirements
The official `vllm/vllm-openai` Linux x86-64 image includes the snapshot
runtime. It still requires a compatible host driver, kernel, and privileges.
Arm64 images omit it. Source installs must set `CRIU_CUDA_PLUGIN_DIR` to the
directory containing `cuda_plugin.so`.

Run snapshot commands with `docker exec` inside a long-lived container. Restore
hands the API server off as a detached process, so a one-shot container would
stop that server when its PID 1 exits. Snapshot preflight requires every
component of the artifact path to be owned by the invoking user or root, with
the directory itself at mode 0700, and the commands below run as root inside
the official image, so the bind-mounted host directory is created with `sudo`
and root ownership. The model downloads in its own step so the captured tree
holds no hub connection, and create runs offline. This example also keeps the
container filesystem and `/dev/shm` namespace stable for the lifetime of the
artifact:

```bash
sudo sysctl kernel.io_uring_disabled=2

snapshot_root="$(pwd)/vllm-snapshots"
sudo install -d -m 0700 -o root -g root "${snapshot_root}"

docker run --detach --name vllm-snapshot \
  --gpus all \
  --privileged \
  --pid=host \
  --ipc=host \
  --network=host \
  --mount "type=bind,source=${snapshot_root},target=/snapshots" \
  --entrypoint sleep \
  vllm/vllm-openai:latest infinity

docker exec vllm-snapshot hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: getting_started/installation/gpu.cuda.inc.md | section: Gpu.Cuda.Inc > Use the custom-built vLLM Docker image**
To run vLLM with the custom-built Docker image:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```

The argument `vllm/vllm-openai` specifies the image to run, and should be replaced with the name of the custom-built image (the `-t` tag from the build command).

Note
    **For version 0.4.1 and 0.4.2 only** - the vLLM docker images under these versions are supposed to be run under the root user since a library under the root user's home directory, i.e. `/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` is required to be loaded during runtime. If you are running the container under a different user, you may need to first change the permissions of the library (and all the parent directories) to allow the user to access it, then run vLLM with environment variable `VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` .

See [Feature x Hardware](../../features/README.md#feature-x-hardware) compatibility matrix for feature support information.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: getting_started/installation/gpu.rocm.inc.md | section: Gpu.Rocm.Inc > Use AMD's Docker Images (Deprecated)
`ARG_PYTORCH_ROCM_ARCH`: Allows to override the gfx architecture values from the base docker image

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 7.0 for MI200 and MI300 series, you can use the default (which build a docker image with `vllm serve` as entrypoint):

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vllm-openai-rocm .
```

To run vLLM with the custom-built Docker image:

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm <args...>
```

The argument `vllm/vllm-openai-rocm` specifies the image to run, and should be replaced with the name of the custom-built image (the `-t` tag from the build command).

To use the docker image as base for development, you can launch it in interactive session through overriding the entrypoint.

Console: Commands
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint bash \
        vllm/vllm-openai-rocm
    ```
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: getting_started/installation/cpu.x86.inc.md | section: Cpu.X86.Inc > Install specific revisions
pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64
```

All available image tags are here: [https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags)

You can run these images via:

```bash
docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai-cpu:latest-x86_64 <args...>
```
[END DOCUMENT 5]
```
</details>

**Answer:**

The non-root `vllm` user has a UID of `2000` in the official Docker image [1].

---

## q001

**Question:** when kv cache runs out and requests get preempted in v1, are they swapped or recomputed by default?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: design/metrics.md | section: Metrics > KV Cache Offloading
Two legacy metrics relate to a "swapped" preemption mode that is no
longer relevant in v1:

- `vllm:num_requests_swapped`
- `vllm:cpu_cache_usage_perc`

In this mode, when a request was preempted (e.g. to make room in KV
cache to complete other requests), kv cache blocks were swapped out to
CPU memory. The `--swap-space` flag has been removed as this feature
is no longer used in V1.

Historically, [vLLM has long supported beam search](https://github.com/vllm-project/vllm/issues/6226). The
SequenceGroup encapsulated the idea of N Sequences which
all shared the same prompt kv blocks. This enabled KV cache block
sharing between requests, and copy-on-write to do branching. CPU
swapping was intended for these beam search like cases.

Later, the concept of prefix caching was introduced, which allowed KV
cache blocks to be shared implicitly. This proved to be a better
option than CPU swapping since blocks can be evicted slowly on demand
and the part of the prompt that was evicted can be recomputed.

SequenceGroup was removed in V1, although a replacement will be
required for "parallel sampling" (`n>1`).
[Beam search was moved out of the core](https://github.com/vllm-project/vllm/issues/8306). There was a
lot of complex code for a very uncommon feature.

In V1, with prefix caching being better (zero over head) and therefore
on by default, the preemption and recompute strategy should work
better.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: features/nixl_connector_usage.md | section: NixlConnector Usage Guide > KV Load Failure Policy
The `kv_load_failure_policy` setting controls how the system handles failures when the decoder instance loads KV cache blocks from the prefiller instance:

- **fail** (default): Immediately fail the request with an error when KV load fails. This prevents performance degradation by avoiding recomputation of prefill work on the decode instance.
- **recompute**: Recompute failed blocks locally on the decode instance. This may cause performance _jitter_ on decode instances as the scheduled prefill will delay and interfere with other decodes. Furthermore, decode instances are typically configured with low-latency optimizations.

Warning
    Using `kv_load_failure_policy="recompute"` can lead to performance degradation in production deployments. When KV loads fail, the decode instance will execute prefill work with decode-optimized configurations, which is inefficient and defeats the purpose of disaggregated prefilling. This also increases tail latency for other ongoing decode requests.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: configuration/optimization.md | section: Optimization and Tuning > Preemption
Due to the autoregressive nature of transformer architecture, there are times when KV cache space is insufficient to handle all batched requests.
In such cases, vLLM can preempt requests to free up KV cache space for other requests. Preempted requests are recomputed when sufficient KV cache space becomes
available again. When this occurs, you may see the following warning:

```text
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

While this mechanism ensures system robustness, preemption and recomputation can adversely affect end-to-end latency.
If you frequently encounter preemptions, consider the following actions:
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: features/kv_offloading_usage.md | section: KV Offloading Usage Guide > Out-of-tree (recommended)
Implement `CachePolicy` (`vllm/v1/kv_offload/cpu/policies/base.py`) in your own package — no vLLM fork or patch required — and point `kv_connector_extra_config` at it directly:

```json
{
  "cpu_bytes_to_use": 10737418240,
  "eviction_policy": "MyCachePolicy",
  "cache_policy_module_path": "my_package.my_module"
}
```

`eviction_policy` is checked against the built-in registry first; if it isn't a registered name, vLLM imports `cache_policy_module_path` and looks up `eviction_policy` as a class name in that module — the same fallback `spec_module_path` provides for a custom `OffloadingSpec`. No import or registration call needs to run before the server starts.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: design/nixl_kv_cache_lease.md | section: NIXL KV Cache Lease Renewal
In disaggregated prefill/decode deployments, the Prefill instance (P) must hold KV cache blocks in GPU memory after completing a prefill, waiting for the Decode instance (D) to read them via RDMA. A mechanism is needed to determine when those blocks can safely be freed when D isn't able to retrieve them. This mechanism was introduced in [PR #41383](https://github.com/vllm-project/vllm/pull/41383).
[END DOCUMENT 5]
```
</details>

**Answer:**

In v1, when KV cache runs out and requests get preempted, they are recomputed by default. The preempted requests are recomputed when sufficient KV cache space becomes available again, rather than being swapped out [1][3].

---

## q058

**Question:** env var to get the same output no matter how requests get batched together

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: benchmarking/sweeps.md | section: Parameter Sweeps > Basic
2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```

4. (Optional) If you would like to vary the settings of `vllm bench serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--bench-params`.

    - Example: Using different input/output lengths for random dataset:

    ```json
    [
        {
            "_benchmark_name": "scenario_A",
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "_benchmark_name": "scenario_B",
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
            "_benchmark_name": "scenario_C",
            "random_input_len": 512,
            "random_output_len": 128
        }
    ]
    ```

5. Set `--output-dir` and optionally `--experiment-name` to control where to save the results.

Example command:

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: deployment/nginx.md | section: Using Nginx > Verify That vLLM Servers Are Ready
```bash
docker logs vllm0 | grep Uvicorn
docker logs vllm1 | grep Uvicorn
```

Both outputs should look like this:

```console
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: serving/online_serving/renderer.md | section: Renderer APIs > Prefill request
Prefill omits `kwargs_data` and sends `mm_metadata` with
`ec_transfer_params` from the encode response:

```json
{
  "token_ids": [151644, 872],
  "features": {
    "mm_hashes": {"image": ["abc123..."]},
    "mm_placeholders": {"image": [{"offset": 0, "length": 256}]},
    "mm_metadata": {
      "image": ["<base64 MultiModalKwargsItem: image_grid_thw only>"]
    }
  },
  "ec_transfer_params": {
    "ec_items": [{"mm_hash": "abc123...", "peer_host": "10.0.0.1"}]
  },
  "sampling_params": {"max_tokens": 64}
}
```
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: features/batch_invariance.md | section: Batch Invariance > Enabling Batch Invariance
Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: serving/online_serving/derenderer.md | section: Derenderer APIs > Streaming
aren't kept between calls, so they go with every chunk including the usage chunk. Both are required when a tool or reasoning parser is configured. `prompt_token_ids` is the `token_ids` of the `GenerateRequest` returned by `/render`.
- **Don't forward `[DONE]`.** It marks the end of the generate stream and isn't a chunk.
[END DOCUMENT 5]
```
</details>

**Answer:**

To achieve batch invariance in vLLM, you need to set the environment variable `VLLM_BATCH_INVARIANT` to `1` by using the command:

```bash
export VLLM_BATCH_INVARIANT=1
``` 

This will ensure that the output remains consistent regardless of how requests are batched together [4].

---

## q040

**Question:** vllm bench isn't working after a plain pip install, what do I need to install?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: deployment/frameworks/streamlit.md | section: Streamlit > Prerequisites
Set up the vLLM environment by installing all required packages:

```bash
pip install vllm streamlit openai
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: getting_started/installation/cpu.x86.inc.md | section: Cpu.X86.Inc > Install specific revisions
--extra-index-url https://download.pytorch.org/whl/cpu
    ```

Build and install vLLM:

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

If you want to develop vLLM, install it in editable mode instead.

```bash
VLLM_TARGET_DEVICE=cpu python3 setup.py develop
```

Optionally, build a portable wheel which you can then install elsewhere:

```bash
VLLM_TARGET_DEVICE=cpu uv build --wheel --no-build-isolation
```

```bash
uv pip install dist/*.whl
```

Console: pip
    ```bash
    VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation
    ```

    ```bash
    pip install dist/*.whl
    ```

Warning: set `LD_PRELOAD`
    Before using vLLM CPU installed via wheels, make sure TCMalloc and Intel OpenMP are installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc, Intel OpenMP is installed with vLLM CPU
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: getting_started/installation/cpu.arm.inc.md | section: Cpu.Arm.Inc > Install specific revisions
pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

Finally, build and install vLLM:

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

If you want to develop vLLM, install it in editable mode instead.

```bash
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

Testing has been conducted on AWS Graviton3 instances for compatibility.

Warning: set `LD_PRELOAD`
    Before use vLLM CPU installed via wheels, make sure TCMalloc is installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

To pull the latest image from Docker Hub:

```bash
docker pull vllm/vllm-openai-cpu:latest-arm64
```

To pull an image with a specific vLLM version:

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-arm64
```

All available image tags are here: [https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags).

You can run these images via:
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: deployment/frameworks/dify.md | section: Dify > Prerequisites
Set up the vLLM environment:

```bash
pip install vllm
```

And install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/).
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: getting_started/installation/cpu.arm.inc.md | section: Cpu.Arm.Inc > Install the latest code
LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides working pre-built Arm CPU wheels for every commit since `v0.11.2` on <https://wheels.vllm.ai/nightly>. For native CPU wheels, this index should be used:

- `https://wheels.vllm.ai/nightly/cpu/vllm`

To install from nightly index, run:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

Console: pip (there's a caveat)

    Using `pip` to install from nightly indices is _not supported_, because `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version. In contrast, `uv` gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

    If you insist on using `pip`, you have to specify the full URL (link address) of the wheel file (which can be obtained from https://wheels.vllm.ai/nightly/cpu/vllm).

    ```bash
    pip install https://wheels.vllm.ai/2f3f441f84bd5b35ec8aa9fcfffb540f107da8a7/vllm-0.23.1rc1.dev901%2Bg2f3f441f8.cpu-cp38-abi3-manylinux_2_34_aarch64.whl --extra-index-url https://download.pytorch.org/whl/cpu # current nightly build (the filename will change!)
    ```
[END DOCUMENT 5]
```
</details>

**Answer:**

To set up the vLLM environment properly, you need to install the required packages using the following command:

```bash
pip install vllm
```

Additionally, if you are using Streamlit, you should also install it along with OpenAI:

```bash
pip install vllm streamlit openai
```

Make sure to follow any additional setup instructions related to your specific environment or framework as needed [1][4].

---

## q029

**Question:** what env vars does the vllm server need to use the ipc weight transfer backend over HTTP?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: training/weight_transfer/README.md | section: Weight Transfer > API Endpoints
When running vLLM as an HTTP server, the following endpoints are available for weight transfer. `HTTPVLLMWeightSyncClient` speaks the first four for you.

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/init_weight_transfer_engine` | POST | Initialize the weight transfer engine with backend-specific info |
| `/start_weight_update` | POST | Start a weight update |
| `/update_weights` | POST | Transfer a batch of weights with backend-specific metadata |
| `/finish_weight_update` | POST | Finish the update and optionally commit its `weight_version` |
| `/update_weight_version` | POST | Update `weight_version` without changing model weights |
| `/weight_info` | GET | Get the latest committed weight version |
| `/pause` | POST | Pause generation before weight sync to handle inflight requests |
| `/resume` | POST | Resume generation after weight sync |
| `/get_world_size` | GET | Get the number of inference workers (useful for NCCL world size calculation) |

Note
    The HTTP weight transfer endpoints require `VLLM_SERVER_DEV_MODE=1` to be set.

The Rust frontend's optional gRPC `Control` service exposes the same pause, sleep, weight-transfer, and weight-version lifecycle for trusted sidecars. The `ServerInfo.rl_capabilities` response reports whether weight transfer and sleep mode were configured. Backend-specific `init_info` and `update_info` remain JSON metadata; model tensors continue to move over the configured NCCL, IPC, sparse-NCCL, or sharded-RDT transport.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: training/weight_transfer/ipc.md | section: IPC Engine > Inference Side
```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(model="my-model", weight_transfer_config=WeightTransferConfig(backend="ipc"))
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "ipc"}'
```

IPC needs no data-plane rendezvous, so `init_transfer_engine` opens no channel —
it only records the `packed` flag the trainer ships at the handshake, which
`receive_weights` then reads. Whether a transfer is packed is therefore never
something you configure on the inference side.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: training/weight_transfer/ipc.md | section: IPC Engine > Trainer Side
```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=IPCTrainerInitInfo(rank=0, packed=False),   # rank 0 is the sender
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
    source=ModuleSource(model),
)

engine.send_weights()   # once per sync
```

`send_weights()` drives `start_weight_update`, the transfer itself, and
`finish_weight_update`, and holds strong references to the IPC-shared copies
until after the post-send barrier — the consumer's views would otherwise dangle.

Any [`VLLMWeightSyncClient`](base.md#vllmweightsyncclient) works here — the
built-in HTTP and Ray clients, or an adapter for your own stack. The engine is
identical either way.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: training/async_rl.md | section: Async Reinforcement Learning > HTTP Endpoints
With `VLLM_SERVER_DEV_MODE=1`, the vLLM HTTP server exposes the same functionality via:

- `POST /pause?mode=keep` - Pause generation
- `POST /resume` - Resume generation
- `POST /abort_requests` - Abort in-flight requests without pausing the scheduler (send `{}` to abort all, or `{"request_ids": [...]}`)
- `GET /weight_info` - Return the latest committed `weight_version`

Note: Data Parallelism
    When using data parallelism with vLLM's **internal load balancer** (i.e. `data_parallel_backend="ray"`), pause and resume are handled automatically across all DP ranks -- a single call is sufficient. When using an **external load balancer** (i.e. multiple independent vLLM instances behind a proxy), you must send pause and resume requests to **every** engine instance individually before and after the weight update.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: training/weight_transfer/m2n.md | section: NCCL M2N Engine > Configuration
```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl_m2n"),
)
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "nccl_m2n"}'
```
[END DOCUMENT 5]
```
</details>

**Answer:**

The vLLM server needs the environment variable `VLLM_SERVER_DEV_MODE=1` to be set in order to use the IPC weight transfer backend over HTTP [1].

---

## q059

**Question:** sweep startup just warns on unknown keys in my params json, how do i make it error out instead?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: benchmarking/sweeps.md | section: Parameter Sweeps > Startup Benchmark
--startup-cmd 'vllm bench startup --model Qwen/Qwen3-0.6B' \
    --serve-params benchmarks/serve_hparams.json \
    --startup-params benchmarks/startup_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

Important
    By default, unsupported parameters in `--serve-params` or `--startup-params` are ignored with a warning.
    Use `--strict-params` to fail fast on unknown keys.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: usage/security.md | section: Security > Usage with a raw request
```json
{
  "model": "meta-llama/Llama-3-8b",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "cache_salt": "per-user-or-per-tenant-secret"
}
```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: benchmarking/sweeps.md | section: Parameter Sweeps > Basic
2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```

4. (Optional) If you would like to vary the settings of `vllm bench serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--bench-params`.

    - Example: Using different input/output lengths for random dataset:

    ```json
    [
        {
            "_benchmark_name": "scenario_A",
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "_benchmark_name": "scenario_B",
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
            "_benchmark_name": "scenario_C",
            "random_input_len": 512,
            "random_output_len": 128
        }
    ]
    ```

5. Set `--output-dir` and optionally `--experiment-name` to control where to save the results.

Example command:

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: benchmarking/sweeps.md | section: Parameter Sweeps > Basic
benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

By default, each parameter combination is benchmarked 3 times to make the results more reliable. You can adjust the number of runs by setting `--num-runs`.

Important
    If both `--serve-params` and `--bench-params` are passed, the script will iterate over the Cartesian product between them.
    You can use `--dry-run` to preview the commands to be run.

    We only start the server once for each `--serve-params`, and keep it running for multiple `--bench-params`.
    Between each benchmark run, we call all `/reset_*_cache` endpoints to get a clean slate for the next run.
    In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.

Note
    You should set `_benchmark_name` to provide a human-readable name for parameter combinations involving many variables.
    This becomes mandatory if the file name would otherwise exceed the maximum path length allowed by the filesystem.

Tip
    You can use the `--resume` option to continue the parameter sweep if an unexpected error occurs, e.g., timeout when connecting to HF Hub.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: features/speculative_decoding/extract_hidden_states.md | section: Hidden State Extraction > Offline usage
Pass per-request options via `extra_args` on `SamplingParams`:

```python
SamplingParams(
    max_tokens=32,
    extra_args={
        "kv_transfer_params": {
            "hidden_states_path": "/tmp/my_output.safetensors",
            "include_output_tokens": True,
        }
    },
)
```
[END DOCUMENT 5]
```
</details>

**Answer:**

Use the `--strict-params` flag to fail fast on unknown keys in your parameters JSON. By default, unsupported parameters in `--serve-params` or `--startup-params` are ignored with a warning [1].

---

## q009

**Question:** requests keep getting preempted because the kv cache fills up, which setting lets vllm grab more gpu memory for cache?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: configuration/optimization.md | section: Optimization and Tuning > Preemption
memory. total_cumulative_preemption_cnt=1
```

While this mechanism ensures system robustness, preemption and recomputation can adversely affect end-to-end latency.
If you frequently encounter preemptions, consider the following actions:

- Increase `gpu_memory_utilization`. vLLM pre-allocates GPU cache using this percentage of memory. By increasing utilization, you can provide more KV cache space.
- Decrease `max_num_seqs` or `max_num_batched_tokens`. This reduces the number of concurrent requests in a batch, thereby requiring less KV cache space.
- Increase `tensor_parallel_size`. This shards model weights across GPUs, allowing each GPU to have more memory available for KV cache. However, increasing this value may cause excessive synchronization overhead.
- Increase `pipeline_parallel_size`. This distributes model layers across GPUs, reducing the memory needed for model weights on each GPU, indirectly leaving more memory available for KV cache. However, increasing this value may cause latency penalties.

You can monitor the number of preemption requests through Prometheus metrics exposed by vLLM. Additionally, you can log the cumulative number of preemption requests by setting `disable_log_stats=False`.

In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: benchmarking/cli.md | section: Benchmark CLI > Load Pattern Configuration
resource limits. During startup, vLLM reports KV cache configuration that directly impacts your load testing parameters:

```text
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
```

Where:

- GPU KV cache size: Total tokens that can be cached across all concurrent requests
- Maximum concurrency: Theoretical maximum concurrent requests for the given `max_model_len`
- Calculation: `max_concurrency = kv_cache_size / max_model_len`

Using KV cache metrics for load pattern configuration:

- For Capacity Planning: Set `--max-concurrency` to 80-90% of the reported maximum to test realistic resource constraints
- For SLA Validation: Use the reported maximum as your SLA limit to ensure compliance testing matches production capacity
- For Realistic Testing: Monitor memory usage when approaching theoretical limits to understand sustainable request rates
- Request rate guidance: Use the KV cache size to estimate sustainable request rates for your specific workload and sequence lengths
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: configuration/conserving_memory.md | section: Conserving Memory > Adjust cache size
If you run out of CPU RAM, try the following options:

- (Multi-modal models only) you can set the size of multi-modal cache by setting `mm_processor_cache_gb` engine argument (default 4 GiB).
- (CPU backend only) you can set the size of KV cache using `VLLM_CPU_KVCACHE_SPACE` environment variable (default 4 GiB).
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: configuration/optimization.md | section: Optimization and Tuning > Preemption
Due to the autoregressive nature of transformer architecture, there are times when KV cache space is insufficient to handle all batched requests.
In such cases, vLLM can preempt requests to free up KV cache space for other requests. Preempted requests are recomputed when sufficient KV cache space becomes
available again. When this occurs, you may see the following warning:

```text
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

While this mechanism ensures system robustness, preemption and recomputation can adversely affect end-to-end latency.
If you frequently encounter preemptions, consider the following actions:
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: deployment/integrations/kthena.md | section: Kthena > 3.2 Apply the `ModelServing`
resources:
                          limits:
                            nvidia.com/gpu: "8"
                            memory: 1124Gi
                            ephemeral-storage: 800Gi
                          requests:
                            ephemeral-storage: 800Gi
                            cpu: 125
                        ports:
                          - containerPort: 8080
                        readinessProbe:
                          tcpSocket:
                            port: 8080
                          initialDelaySeconds: 15
                          periodSeconds: 10
                        volumeMounts:
                          - mountPath: /dev/shm
                            name: dshm
                    volumes:
                    - name: dshm
                      emptyDir:
                        medium: Memory
                        sizeLimit: 15Gi
                workerReplicas: 1
                workerTemplate:
                  spec:
                    containers:
                      - name: worker
                        image: vllm/vllm-openai:latest
                        command:
                          - sh
                          - -c
                          - "vllm serve meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 1 --master-addr $(ENTRY_ADDRESS) --distributed-executor-backend mp --headless"
                        resources:
                          limits:
                            nvidia.com/gpu: "8"
[END DOCUMENT 5]
```
</details>

**Answer:**

To allow vLLM to grab more GPU memory for the KV cache, you can increase `gpu_memory_utilization`. This setting pre-allocates GPU cache using a higher percentage of memory, providing more KV cache space. Additionally, you might consider increasing `tensor_parallel_size`, which shards model weights across GPUs, allowing each GPU to have more memory available for KV cache. However, be aware that increasing this value may cause excessive synchronization overhead [1][4].

---

## q044

**Question:** i want vllm to pull models from modelscope instead of hugging face

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: models/supported_models.md | section: Supported Models > ModelScope
To use models from [ModelScope](https://www.modelscope.cn) instead of Hugging Face Hub, set an environment variable:

```shell
export VLLM_USE_MODELSCOPE=True
```

And use with `trust_remote_code=True`.

```python
from vllm import LLM

llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)

# For generative models (runner=generate) only
output = llm.generate("Hello, my name is")
print(output)

# For pooling models (runner=pooling) only
output = llm.encode("Hello, my name is")
print(output)
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: models/supported_models.md | section: Supported Models > Hugging Face Hub
By default, vLLM loads models from [Hugging Face (HF) Hub](https://huggingface.co/models). To change the download path for models, you can set the `HF_HOME` environment variable; for more details, refer to [their official documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome).

To determine whether a given model is natively supported, you can check the `config.json` file inside the HF repository.
If the `"architectures"` field contains a model architecture listed below, then it should be natively supported.

Models do not _need_ to be natively supported to be used in vLLM.
The [Transformers modeling backend](#transformers) enables you to run models directly using their Transformers implementation (or even remote code on the Hugging Face Model Hub!).

Tip
    The easiest way to check if your model is really supported at runtime is to run the program below:

    ```python
    from vllm import LLM

    # For generative models (runner=generate) only
    llm = LLM(model=..., runner="generate")  # Name or path of your model
    output = llm.generate("Hello, my name is")
    print(output)

    # For pooling models (runner=pooling) only
    llm = LLM(model=..., runner="pooling")  # Name or path of your model
    output = llm.encode("Hello, my name is")
    print(output)
    ```

    If vLLM successfully returns text (for generative models) or hidden states (for pooling models), it indicates that your model is supported.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: models/supported_models.md | section: Supported Models > Hugging Face Hub
or path of your model
    output = llm.encode("Hello, my name is")
    print(output)
    ```

    If vLLM successfully returns text (for generative models) or hidden states (for pooling models), it indicates that your model is supported.

Otherwise, please refer to [Adding a New Model](../contributing/model/README.md) for instructions on how to implement your model in vLLM.
Alternatively, you can [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose) to request vLLM support.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: models/supported_models.md | section: Supported Models > Custom models
If a model is neither supported natively by vLLM nor Transformers, it can still be used in vLLM!

For a model to be compatible with the Transformers modeling backend for vLLM it must:

- be a Transformers compatible custom model (see [Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models)):
    - The model directory must have the correct structure (e.g. `config.json` is present).
    - `config.json` must contain `auto_map.AutoModel`.
- be a Transformers modeling backend for vLLM compatible model (see [Writing custom models](#writing-custom-models)):
    - Customisation should be done in the base model (e.g. in `MyModel`, not `MyModelForCausalLM`).

If the compatible model is:

- on the Hugging Face Model Hub, simply set `trust_remote_code=True` for [offline-inference](../serving/offline_inference.md) or `--trust-remote-code` for the [online serving](../serving/online_serving/README.md).
- in a local directory, simply pass directory path to `model=<MODEL_DIR>` for [offline-inference](../serving/offline_inference.md) or `vllm serve <MODEL_DIR>` for the [online serving](../serving/online_serving/README.md).

This means that, with the Transformers modeling backend for vLLM, new models can be used before they are officially supported in Transformers or vLLM!
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: design/huggingface_integration.md | section: Integration with Hugging Face
describes how vLLM integrates with Hugging Face libraries. We will explain step by step what happens under the hood when we run `vllm serve`.

Let's say we want to serve the popular Qwen model by running `vllm serve Qwen/Qwen2-7B`.

1. The `model` argument is `Qwen/Qwen2-7B`. vLLM determines whether this model exists by checking for the corresponding config file `config.json`. See this [code snippet](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182) for the implementation. Within this process:
    - If the `model` argument corresponds to an existing local path, vLLM will load the config file directly from this path.
    - If the `model` argument is a Hugging Face model ID consisting of a username and model name, vLLM will first try to use the config file from the Hugging Face local cache, using the `model` argument as the model name and the `--revision` argument as the revision. See [their website](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) for more information on how the Hugging Face cache works.
    - If the `model` argument is a Hugging Face model ID but it is not found in the cache, vLLM will download the config file from the Hugging Face model hub. Refer to [this function](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91) for the implementation. The input arguments include the `model` argument as the model name, the `--revision` argument as the revision, and the environment variable `HF_TOKEN` as the token to access the model hub. In our case, vLLM will download the [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) file.
[END DOCUMENT 5]
```
</details>

**Answer:**

To use models from ModelScope instead of Hugging Face Hub, set the environment variable:

```shell
export VLLM_USE_MODELSCOPE=True
```

Additionally, you need to use `trust_remote_code=True` when initializing the `LLM` class:

```python
llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)
``` 

This configuration allows vLLM to pull models from ModelScope [1].

---

## q022

**Question:** model fails to load because its config.json has no architectures field, how do i fix that?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: configuration/model_resolution.md | section: Model Resolution
vLLM loads HuggingFace-compatible models by inspecting the `architectures` field in `config.json` of the model repository
and finding the corresponding implementation that is registered to vLLM.
Nevertheless, our model resolution may fail for the following reasons:

- The `config.json` of the model repository lacks the `architectures` field.
- Unofficial repositories refer to a model using alternative names which are not recorded in vLLM.
- The same architecture name is used for multiple models, creating ambiguity as to which model should be loaded.

To fix this, explicitly specify the model architecture by passing `config.json` overrides to the `hf_overrides` option.
For example:

```python
from vllm import LLM

llm = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

Our [list of supported models](../models/supported_models.md) shows the model architectures that are recognized by vLLM.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: usage/troubleshooting.md | section: Troubleshooting > Model failed to be inspected
If you see an error like:

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] failed to be inspected. Please check the logs for more details.
```

It means that vLLM failed to import the model file.
Usually, it is related to missing dependencies or outdated binaries in the vLLM build.
Please read the logs carefully to determine the root cause of the error.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: usage/troubleshooting.md | section: Troubleshooting > Model not supported
If you see an error like:

```text
Traceback (most recent call last):
...
  File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
    for arch in architectures:
TypeError: 'NoneType' object is not iterable
```

or:

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] are not supported for now. Supported architectures: [...]
```

But you are sure that the model is in the [list of supported models](../models/supported_models.md), there may be some issue with vLLM's model resolution. In that case, please follow [these steps](../configuration/model_resolution.md) to explicitly specify the vLLM implementation for the model.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: serving/integrations/codex.md | section: Codex > Troubleshooting
**Connection refused**: Ensure vLLM is running and accessible at the specified URL. Check that the port matches and that `base_url` includes the `/v1` path suffix.

**Tool calls not working**: Verify that your model supports tool calling and that you've enabled it with the correct `--tool-call-parser` flag. See [Tool Calling](../../features/tool_calling.md).

**Model not found**: Ensure the `model` field in `~/.codex/config.toml` matches the `--served-model-name` you passed to vLLM.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: deployment/frameworks/helm.md | section: Helm > Using S3 Model Download (Default)
```yaml
extraInit:
  modelDownload:
    enabled: true
  pvcStorage: "10Gi"
  s3modelpath: "models/llama-7b"
```
[END DOCUMENT 5]
```
</details>

**Answer:**

To fix the issue of a model failing to load due to the absence of the `architectures` field in its `config.json`, you can explicitly specify the model architecture by passing `config.json` overrides to the `hf_overrides` option. For example:

```python
from vllm import LLM

llm = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

This will help vLLM correctly identify the model architecture to use [1].

---

## q060

**Question:** what do i put in my vllm serve config.yaml to load weights with the run:ai model streamer?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: models/extensions/runai_model_streamer.md | section: Loading models with Run:ai Model Streamer
Run:ai Model Streamer is a library to read tensors in concurrency, while streaming it to GPU memory.
Further reading can be found in [Run:ai Model Streamer Documentation](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md).

vLLM supports loading weights in Safetensors format using the Run:ai Model Streamer.
You first need to install vLLM RunAI optional dependency:

```bash
pip3 install vllm[runai]
```

To run it as an OpenAI-compatible server, add the `--load-format runai_streamer` flag:

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer
```

To run model from AWS S3 object store run:

```bash
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

To run model from Google Cloud Storage run:

```bash
vllm serve gs://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

To run model from Azure Blob Storage run:

```bash
AZURE_STORAGE_ACCOUNT_NAME=<account> \
vllm serve az://<container>/<model-path> \
    --load-format runai_streamer
```

Authentication uses `DefaultAzureCredential`, which supports `az login`, managed identity, environment variables (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`), and other methods.

To run model from a S3 compatible object store run:

```bash
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
AWS_EC2_METADATA_DISABLED=true \
AWS_ENDPOINT_URL=https://storage.googleapis.com \
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: models/extensions/runai_model_streamer.md | section: Loading models with Run:ai Model Streamer > Sharded Model Loading
vLLM also supports loading sharded models using Run:ai Model Streamer. This is particularly useful for large models that are split across multiple files. To use this feature, use the `--load-format runai_streamer_sharded` flag:

```bash
vllm serve /path/to/sharded/model --load-format runai_streamer_sharded
```

The sharded loader expects model files to follow the same naming pattern as the regular sharded state loader: `model-rank-{rank}-part-{part}.safetensors`. You can customize this pattern using the `pattern` parameter in `--model-loader-extra-config`:

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"pattern":"custom-model-rank-{rank}-part-{part}.safetensors"}'
```

To create sharded model files, you can use the script provided in [examples/features/sharded_state/save_sharded_state_offline.py](../../../examples/features/sharded_state/save_sharded_state_offline.py). This script demonstrates how to save a model in the sharded format that is compatible with the Run:ai Model Streamer sharded loader.

The sharded loader supports all the same tunable parameters as the regular Run:ai Model Streamer, including `concurrency` and `memory_limit`. These can be configured in the same way:

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: models/extensions/instanttensor.md | section: Loading Model Weights with InstantTensor > Use InstantTensor in vLLM
Add `--load-format instanttensor` as a command-line argument.

For example:

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format instanttensor
```
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: deployment/integrations/production-stack.md | section: Production stack > (Advanced) Configuring vLLM production stack
The core vLLM production stack configuration is managed with YAML. Here is the example configuration used in the installation above:

Code: Yaml

    ```yaml
    servingEngineSpec:
      runtimeClassName: ""
      modelSpec:
      - name: "opt125m"
        repository: "vllm/vllm-openai"
        tag: "latest"
        modelURL: "facebook/opt-125m"

        replicaCount: 1

        requestCPU: 6
        requestMemory: "16Gi"
        requestGPU: 1

        pvcStorage: "10Gi"
    ```

In this YAML configuration:

* **`modelSpec`** includes:
    * `name`: A nickname that you prefer to call the model.
    * `repository`: Docker repository of vLLM.
    * `tag`: Docker image tag.
    * `modelURL`: The LLM model that you want to use.
* **`replicaCount`**: Number of replicas.
* **`requestCPU` and `requestMemory`**: Specifies the CPU and memory resource requests for the pod.
* **`requestGPU`**: Specifies the number of GPUs required.
* **`pvcStorage`**: Allocates persistent storage for the model.

Note
    If you intend to set up two pods, please refer to this [YAML file](https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-01-2pods-minimal-example.yaml).

Tip
    vLLM production stack offers many more features (*e.g.* CPU offloading and a wide range of routing algorithms). Please check out these [examples and tutorials](https://github.com/vllm-project/production-stack/tree/main/tutorials) and our [repo](https://github.com/vllm-project/production-stack) for more details!
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: models/extensions/runai_model_streamer.md | section: Loading models with Run:ai Model Streamer > Sharded Model Loading
and `memory_limit`. These can be configured in the same way:

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```

Note
    The sharded loader is particularly efficient for tensor or pipeline parallel models where each worker only needs to read its own shard rather than the entire checkpoint.
[END DOCUMENT 5]
```
</details>

**Answer:**

To load weights with the Run:ai Model Streamer in your `vllm serve` configuration, you need to specify the `--load-format runai_streamer` flag in your command. However, the provided excerpts do not contain specific YAML configuration details for this purpose. 

You can run the command as follows:

```bash
vllm serve /path/to/model --load-format runai_streamer
```

For sharded models, you would use:

```bash
vllm serve /path/to/sharded/model --load-format runai_streamer_sharded
```

The excerpts do not provide a direct YAML configuration example for this specific case.

---

## q033

**Question:** pdb breakpoints in vllm just throw BdbQuit, how do I debug?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: usage/troubleshooting.md | section: Troubleshooting > Breakpoints
Setting normal `pdb` breakpoints may not work in vLLM's codebase if they are executed in a subprocess. You will experience something like:

``` text
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 100, in trace_dispatch
    return self.dispatch_line(frame)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 125, in dispatch_line
    if self.quitting: raise BdbQuit
                      ^^^^^^^^^^^^^
bdb.BdbQuit
```

One solution is using [forked-pdb](https://github.com/Lightning-AI/forked-pdb). Install with `pip install fpdb` and set a breakpoint with something like:

``` python
__import__('fpdb').ForkedPdb().set_trace()
```

Another option is to disable multiprocessing entirely, with the `VLLM_ENABLE_V1_MULTIPROCESSING` environment variable.
This keeps the scheduler in the same process, so you can use stock `pdb` breakpoints:

``` python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: design/debug_vllm_compile.md | section: How to debug the vLLM-torch.compile integration > Debugging TorchDynamo
vLLM requires model code be capturable into a full graph via TorchDynamo (torch.compile's frontend).
TorchDynamo does not support all of Python. It will error (in fullgraph mode) if it cannot support
a feature (this is sometimes known as a graph break).

If you encounter a graph break, please [open an issue to pytorch/pytorch](https://github.com/pytorch/pytorch) so the PyTorch devs can prioritize.
Then, try your best to rewrite the code to avoid the graph break.
For more information, see this [Dynamo guide](https://docs.pytorch.org/docs/stable/compile/programming_model.dynamo_core_concepts.html).
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: design/debug_vllm_compile.md | section: How to debug the vLLM-torch.compile integration > Inductor runtime assertions
By default (on torch < 2.12), vLLM disables Inductor's runtime assertions
(`assert_size_stride`, `assert_alignment`) to avoid ~2ms overhead per forward
pass on large models. Setting `VLLM_LOGGING_LEVEL=DEBUG` automatically
re-enables them so debugging sessions get full shape/stride validation:

```sh
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model>
```

You can also override them explicitly via `--compilation-config`:

```sh
vllm serve <model> -cc.inductor_compile_config='{"size_asserts": true, "alignment_asserts": true, "scalar_asserts": true}'
```

On torch >= 2.12, PyTorch uses an efficient assert-once strategy and these
flags are no longer suppressed by vLLM.

To debug if TorchInductor is at fault, you can disable it by passing `backend='eager'`
to the compilation config:

```sh
# online
vllm serve -cc.backend=eager
```

```py
# offline
LLM(compilation_config=CompilationConfig(backend='eager'))
```

If Inductor is at fault, [file a bug to PyTorch](https://github.com/pytorch/pytorch).
If you're feeling adventurous, you can debug the triton kernels in the Inductor output code
(that you can locate via using tlparse).

![tlparse example](../assets/design/debug_vllm_compile/tlparse_inductor.png)

You can also use `TORCH_LOGS=output_code <command>` to print the Inductor output code.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: usage/troubleshooting.md | section: Troubleshooting
This document outlines some troubleshooting strategies you can consider. If you think you've discovered a bug, please [search existing issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue) first to see if it has already been reported. If not, please [file a new issue](https://github.com/vllm-project/vllm/issues/new/choose), providing as much relevant information as possible.

Note
    Once you've debugged a problem, remember to turn off any debugging environment variables defined, or simply start a new shell to avoid being affected by lingering debugging settings. Otherwise, the system might be slow with debugging functionalities left activated.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: design/debug_vllm_compile.md | section: How to debug the vLLM-torch.compile integration > Debugging Dynamic Shape full graph capture
vLLM requires that the model's forward pass be capturable into a full graph that is dynamic
on the batch size (i.e. the number of tokens). It (by default) compiles this one graph into
one artifact and uses this artifact for all batch sizes.

If your code cannot be captured with Dynamic Shapes, you may see silent incorrectness,
loud errors, or CUDA illegal memory accesses. For example, the following is not
capturable into a single graph:

```py
if data.size[0] % 128 == 0:
    foo(...)
else:
    bar(...)
```

This problem is easy to diagnose. Use tlparse and click on `compilation_metrics`:
it will tell you symbolic constraints on the batch size. If there is any constraint
that restricts the batch sizes, then we've got a problem.

![Bad tlparse example](../assets/design/debug_vllm_compile/dynamic_shapes.png)

To avoid this, please either:

1. avoid branching on the number of tokens
2. wrap the branching logic into a custom operator. TorchDynamo does not
trace into custom operators.
[END DOCUMENT 5]
```
</details>

**Answer:**

You can use `forked-pdb` as a solution. Install it with `pip install fpdb` and set a breakpoint using:

```python
__import__('fpdb').ForkedPdb().set_trace()
```

Alternatively, you can disable multiprocessing entirely by setting the environment variable:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

This allows you to use stock `pdb` breakpoints without encountering the `BdbQuit` error [1].

---

## q057

**Question:** I want to send a trickle of tiny side requests during a benchmark to see how much the main load slows them down, which option does that?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: benchmarking/cli.md | section: Benchmark CLI > Probe Requests
The benchmark tool also supports sending probe requests alongside the main
workload. This can be useful for measuring how the main workload affects
unrelated traffic sharing the server, e.g. a few requests with large images
stalling a concurrent lightweight request while their multimodal preprocessing
occupies the frontend.

Setting `--probe-request-rate` to a positive value sends single-token text-only
probe requests at that rate (requests per second) alongside the main workload.
Probes bypass `--max-concurrency` and their latency is reported separately, so
the probe percentiles directly measure the interference that the main workload
inflicts on unrelated requests.

```bash
vllm bench serve \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name random-mm \
    --random-mm-bucket-config '{(2048, 2048, 1): 1.0}' \
    --request-rate 4 \
    --probe-request-rate 20
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: benchmarking/cli.md | section: Benchmark CLI > Running With Ramp-Up Request Rate
The benchmark tool also supports ramping up the request rate over the
duration of the benchmark run. This can be useful for stress testing the
server or finding the maximum throughput that it can handle, given some latency budget.

Two ramp-up strategies are supported:

- `linear`: Increases the request rate linearly from a start value to an end value.
- `exponential`: Increases the request rate exponentially.

The following arguments can be used to control the ramp-up:

- `--ramp-up-strategy`: The ramp-up strategy to use (`linear` or `exponential`).
- `--ramp-up-start-rps`: The request rate at the beginning of the benchmark.
- `--ramp-up-end-rps`: The request rate at the end of the benchmark.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: benchmarking/cli.md | section: Benchmark CLI > Load Pattern Configuration
vLLM's benchmark serving script provides sophisticated load pattern simulation capabilities through three key parameters that control request generation and concurrency behavior:

##### Load Pattern Control Parameters

- `--request-rate`: Controls the target request generation rate (requests per second). Set to `inf` for maximum throughput testing or finite values for controlled load simulation.
- `--burstiness`: Controls traffic variability using a Gamma distribution (range: > 0). Lower values create bursty traffic, higher values create uniform traffic.
- `--max-concurrency`: Limits concurrent outstanding requests. If this argument is not provided, concurrency is unlimited. Set a value to simulate backpressure. When set, include `client_queue_time` in `--percentile-metrics` to report time spent waiting for the benchmark client's concurrency limit. With a finite `--request-rate`, `e2el_including_client_queue` reports schedule-relative end-to-end latency; it is omitted for `--request-rate=inf`, where all requests arrive at benchmark start.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: benchmarking/cli.md | section: Benchmark CLI > 🚀 Online Benchmark
\
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total token throughput (tok/s):          619.85
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: cli/README.md | section: vLLM CLI Guide > bench
Run benchmark tests for latency online serving throughput and offline inference throughput.

To use benchmark commands, please install with extra dependencies using `pip install vllm[bench]`.

Available Commands:

```bash
vllm bench {latency, serve, throughput}
```
[END DOCUMENT 5]
```
</details>

**Answer:**

To send a trickle of tiny side requests during a benchmark and measure how much the main load slows them down, you should set the `--probe-request-rate` to a positive value. This will send single-token text-only probe requests at that rate alongside the main workload, allowing you to measure the interference that the main workload inflicts on unrelated requests [1].

---

## q053

**Question:** how do i add a new lora adapter to a running vllm server without restarting it?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: features/lora.md | section: LoRA Adapters > Dynamically serving LoRA Adapters
In addition to serving LoRA adapters at server startup, the vLLM server supports dynamically configuring LoRA adapters at runtime through dedicated API endpoints and plugins. This feature can be particularly useful when the flexibility to change models on-the-fly is needed.

Warning
    This feature comes with security risks. It should not be used in production unless it is an isolated, fully trusted environment.

To enable dynamic LoRA configuration, ensure that the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
is set to `True`.

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: design/lora_resolver_plugins.md | section: LoRA Resolver Plugins > Setup Steps
1. **Create the LoRA adapter storage directory**:
   ```bash
   mkdir -p /path/to/lora/adapters
   ```

2. **Set environment variables**:
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   export VLLM_PLUGINS=lora_filesystem_resolver
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

3. **Start vLLM server**:
   Your base model can be `meta-llama/Llama-2-7b-hf`. Please make sure you set up the Hugging Face token in your env var `export HF_TOKEN=xxx235`.
   ```bash
   vllm serve your-base-model \
       --enable-lora
   ```
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: features/lora.md | section: LoRA Adapters > Using Plugins
plugin to load LoRA adapters from repositories on Hugging Face Hub](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)
To enable either of these resolvers, you must `set VLLM_ALLOW_RUNTIME_LORA_UPDATING` to True.

- To leverage a local directory, set `VLLM_PLUGINS` to include `lora_filesystem_resolver` and set `VLLM_LORA_RESOLVER_CACHE_DIR` to a local directory. When vLLM receives a request using a LoRA adapter `foobar`,
it will first look in the local directory for a directory `foobar`, and attempt to load the contents of that directory as a LoRA adapter. If successful, the request will complete as normal and that adapter will then be available for normal use on the server.
- To leverage repositories on Hugging Face Hub, set `VLLM_PLUGINS` to include `lora_hf_hub_resolver` and set `VLLM_LORA_RESOLVER_HF_REPO_LIST` to a comma separated list of repository IDs on Hugging Face Hub. When vLLM receives a request for the LoRA adapter `my/repo/subpath`, it will download the adapter at the `subpath` of `my/repo` if it exists and contains an `adapter_config.json`, then build a request to the cached dir for the adapter, similar to the `lora_filesystem_resolver`. Please note that enabling remote downloads is insecure and not intended for use in production environments.

Alternatively, follow these example steps to implement your own plugin:

1. Implement the LoRAResolver interface.

    Code: Example of a simple S3 LoRAResolver implementation
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: usage/security.md | section: Security > Dynamic LoRA Loading
vLLM supports dynamically loading and unloading LoRA adapters at runtime via the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` API endpoints. This functionality is **not enabled by default** — it requires both `--enable-lora` and the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` to be set.

**Warning:** Dynamic LoRA loading is not a secure operation and should not be enabled in deployments exposed to untrusted clients. If you must enable dynamic LoRA loading, restrict access to the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints to trusted administrators only, using a reverse proxy or network-level access controls. Do not expose these endpoints to end users. For details on configuring LoRA adapters, see the [LoRA Adapters documentation](../features/lora.md).
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: features/lora.md | section: LoRA Adapters > Using Plugins
Alternatively, you can use the LoRAResolver plugin to dynamically load LoRA adapters. LoRAResolver plugins enable you to load LoRA adapters from both local and remote sources such as local file system and S3. On every request, when there's a new model name that hasn't been loaded yet, the LoRAResolver will try to resolve and load the corresponding LoRA adapter.

You can set up multiple LoRAResolver plugins if you want to load LoRA adapters from different sources. For example, you might have one resolver for local files and another for S3 storage. vLLM will load the first LoRA adapter that it finds.

You can either install existing plugins or implement your own. By default, vLLM comes with a [resolver plugin to load LoRA adapters from a local directory, as well as a resolver plugin to load LoRA adapters from repositories on Hugging Face Hub](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)
To enable either of these resolvers, you must `set VLLM_ALLOW_RUNTIME_LORA_UPDATING` to True.
[END DOCUMENT 5]
```
</details>

**Answer:**

To add a new LoRA adapter to a running vLLM server without restarting it, you need to enable dynamic LoRA configuration. Set the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING` to `True`:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

Then, ensure that the server is started with the `--enable-lora` flag. You can dynamically load the LoRA adapter using the `/v1/load_lora_adapter` API endpoint [1][4]. Note that this feature comes with security risks and should not be used in production unless in a trusted environment [1].

---

## q007

**Question:** when tool_choice is none I don't want the tool schemas stuffed into the prompt at all, what option does that?

<details><summary>Excerpts the model saw</summary>

```text
[BEGIN DOCUMENT 1] source: features/tool_calling.md | section: Tool Calling > None Function Calling
vLLM supports the `tool_choice='none'` option in the chat completion API. When this option is set, the model will not generate any tool calls and will respond with regular text content only, even if tools are defined in the request.

Note
    When tools are specified in the request, vLLM includes tool definitions in the prompt by default, regardless of the `tool_choice` setting. To exclude tool definitions when `tool_choice='none'`, use the `--exclude-tools-when-tool-choice-none` option.
[END DOCUMENT 1]

[BEGIN DOCUMENT 2] source: features/tool_calling.md | section: Tool Calling > Server-Side Strictness Floor
Most OpenAI-compatible clients and agent frameworks never set `strict` on their tools, so with `tool_choice="auto"` the model generates tool calls without any grammar and malformed markup can leak into the response. The `--tool-strict-level` option lets the server operator raise the floor for every request that carries tools, independently of what the client declares:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Follow the request's tool choice and per-tool strictness. Required/named choices activate structural tags; `tool_choice="auto"` activates them when at least one tool sets `strict: true`. |
| `function` | Constrain the tool-call envelope (markup and the function name) for every request with tools, leaving argument contents free unless the client marked the tool `strict: true`. |
| `parameter` | Additionally pin argument schemas for every tool, as if every tool had `strict: true`. |

```bash
vllm serve ... --tool-strict-level function
```

The floor never relaxes a constraint the request would already receive: tools the client marked `strict: true` keep their schemas at every level. With `tool_choice="auto"`, the grammar does not force a tool call; a plain text response stays valid. `VLLM_ENFORCE_STRICT_TOOL_CALLING=false` disables structural tags entirely and takes precedence over this option.
[END DOCUMENT 2]

[BEGIN DOCUMENT 3] source: features/tool_calling.md | section: Tool Calling
vLLM currently supports named function calling, as well as the `auto`, `required` (as of `vllm>=0.8.3`), and `none` options for the `tool_choice` field in the chat completion API.
[END DOCUMENT 3]

[BEGIN DOCUMENT 4] source: features/tool_calling.md | section: Tool Calling > Constrained Decoding Behavior
Structural-tag parsers resolve call obligation, grammar activation, and argument-schema enforcement separately. With structural-tag enforcement enabled:

| `tool_choice` value | Call obligation | Structural-tag activation |
| --- | --- | --- |
| Named function | Call the selected function | Always |
| `"required"` | Produce at least one tool call | Always |
| `"auto"` | Tool calls are optional | When at least one tool sets `strict: true`, or `--tool-strict-level` is `function` or `parameter` |
| `"none"` | Tool calling is disabled | Disabled |

When a structural tag applies, each tool's declared parameter schema is enforced only when that tool sets `strict: true` or the server uses `--tool-strict-level parameter`. Tools with omitted or false `strict` receive broad argument-syntax constraints at levels `auto` and `function`, including for required and named calls.

For parsers using schema-derived JSON constraints, required and named calls continue to enforce the declared parameter schemas.
[END DOCUMENT 4]

[BEGIN DOCUMENT 5] source: features/tool_calling.md | section: Tool Calling > Required Function Calling
vLLM supports the `tool_choice='required'` option in the chat completion API. Similar to the named function calling, it also uses structured outputs, so this is enabled by default and will work with any supported model. However, support for alternative decoding backends are on the [roadmap](../usage/v1_guide.md#features) for the V1 engine.

When tool_choice='required' is set, the model is guaranteed to generate one or more tool calls based on the specified tool list in the `tools` parameter. The number of tool calls depends on the user's query. The output format strictly follows the schema defined in the `tools` parameter.
[END DOCUMENT 5]
```
</details>

**Answer:**

To exclude tool definitions when `tool_choice='none'`, use the `--exclude-tools-when-tool-choice-none` option [1].
