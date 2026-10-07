# Golden set evidence

For review: each question with the exact doc text that proves its answer.
Questions q046–q060 are the hold-out set: don't tune prompts or chunking on them.

### q001 · factual
**Q:** when kv cache runs out and requests get preempted in v1, are they swapped or recomputed by default?  
**A:** They get recomputed. RECOMPUTE is the default preemption mode in V1, not SWAP, because it has lower overhead there.  
**Source:** `configuration/optimization.md` › Preemption  
**Evidence:** In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture.

### q002 · factual
**Q:** which eviction policy does the OffloadingConnector CPU tier use by default?  
**A:** It defaults to `lru`; `arc` is the other built-in option, or you can plug in a custom CachePolicy.  
**Source:** `features/kv_offloading_usage.md` › `kv_connector_extra_config` Reference  
**Evidence:** | `eviction_policy` | no | `lru` | both | Primary tier policy: built-in `lru`/`arc`, or a custom `CachePolicy` name

### q003 · factual
**Q:** which -O optimization level does vllm use if i don't pass one?  
**A:** -O2 is the default. It adds extra compilation ranges and fusions and uses FULL_AND_PIECEWISE cudagraphs.  
**Source:** `configuration/optimization.md;design/optimization_levels.md` › Optimization Levels  
**Evidence:** - `-O2`: Default optimization. Additional compilation ranges, additional fusions, FULL_AND_PIECEWISE cudagraphs.

### q004 · how_to
**Q:** is there a way to switch off qwen3 thinking for every request without changing client code?  
**A:** Yes. Set a server-wide default with `--default-chat-template-kwargs '{"enable_thinking": false}'` (alongside `--reasoning-parser qwen3`). Clients can still override it per request via chat_template_kwargs.  
**Source:** `features/reasoning_outputs.md` › Disabling Thinking Mode by Default  
**Evidence:** For models like Qwen3 where thinking is enabled by default, you can disable it server-wide:

### q005 · multi_hop
**Q:** if I turn off v1 multiprocessing so pdb works, does that mess with my script's random state?  
**A:** Yes. Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 (the pdb workaround) puts the workers in your process, so vLLM changes the random state of the code that builds the LLM class.  
**Source:** `usage/troubleshooting.md;usage/reproducibility.md` › Breakpoints  
**Evidence:** Another option is to disable multiprocessing entirely, with the `VLLM_ENABLE_V1_MULTIPROCESSING` environment variable. || Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` will change the random state of user code

### q006 · factual
**Q:** what happens to in-flight requests by default when I call pause_generation?  
**A:** The default mode is "abort": all in-flight requests are aborted immediately and partial results are returned.  
**Source:** `training/async_rl.md` › pause_generation  
**Evidence:** | `"abort"` | Abort all in-flight requests immediately and return partial results (default) |

### q007 · config_flag
**Q:** when tool_choice is none I don't want the tool schemas stuffed into the prompt at all, what option does that?  
**A:** Start the server with --exclude-tools-when-tool-choice-none. Without it, vLLM includes tool definitions in the prompt whatever tool_choice is set to.  
**Source:** `features/tool_calling.md` › None Function Calling  
**Evidence:** To exclude tool definitions when `tool_choice='none'`, use the `--exclude-tools-when-tool-choice-none` option.

### q008 · factual
**Q:** what UID does the non-root vllm user have in the official docker image?  
**A:** The built-in `vllm` user is UID 2000 with GID 0, so you run it with `--user 2000:0`.  
**Source:** `deployment/docker.md` › Run as a non-root user  
**Evidence:** It is also prepared to run as the built-in `vllm` user
(UID 2000, GID 0):

### q009 · config_flag
**Q:** requests keep getting preempted because the kv cache fills up, which setting lets vllm grab more gpu memory for cache?  
**A:** Raise `gpu_memory_utilization`, the fraction of GPU memory vLLM pre-allocates for the cache. Increasing `tensor_parallel_size` or lowering `max_num_seqs` / `max_num_batched_tokens` also helps.  
**Source:** `configuration/optimization.md` › Preemption  
**Evidence:** Increase `gpu_memory_utilization`. vLLM pre-allocates GPU cache using this percentage of memory.

### q010 · unanswerable
**Q:** how much throughput do I lose by turning on text watermarking?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped whole docs folder for 'watermark' (only features/watermarking.md); grepped that page for overhead|latency|throughput|slowdown|benchmark and grepped all watermark lines for latency/overhead/tokens/s/% - no hits. No performance-cost numbers are given.

### q011 · how_to
**Q:** how do i check if NCCL is actually using GPUDirect RDMA across nodes?  
**A:** Start vLLM with `NCCL_DEBUG=TRACE vllm serve ...` and look in the logs: `[send] via NET/IB/GDRDMA` means InfiniBand with GPUDirect RDMA, while `[send] via NET/Socket` means it fell back to plain TCP.  
**Source:** `serving/parallelism_scaling.md` › Enabling GPUDirect RDMA  
**Evidence:** If you find `[send] via NET/IB/GDRDMA` in the logs, then NCCL is using InfiniBand with GPUDirect RDMA, which *is* efficient.

### q012 · multi_hop
**Q:** ray says 'No available node types can fulfill resource request' even though I have enough GPUs, what IP should I give vllm?  
**A:** Set `VLLM_HOST_IP` (a different value on each node) so vLLM and Ray agree on the node's IP, and use an address on a private network segment since that traffic is unencrypted.  
**Source:** `serving/distributed_troubleshooting.md;serving/parallelism_scaling.md` › No available node types can fulfill resource request  
**Evidence:** Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP` in [examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) (with a different value on each node). || For security, set `VLLM_HOST_IP` to an address on a private network segment.

### q013 · multi_hop
**Q:** serving QwQ-32B with reasoning plus function calling, which parsers do I pass?  
**A:** Use `--reasoning-parser deepseek_r1` for QwQ-32B, and `--tool-call-parser hermes` (with --enable-auto-tool-choice) for tool calls, since Qwen models use Hermes-style tool use.  
**Source:** `features/reasoning_outputs.md;features/tool_calling.md` › Supported Models  
**Evidence:** | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | `deepseek_r1` | `json`, `regex` | ✅ | || you can use the `hermes` parser to enable tool calls for Qwen models

### q014 · how_to
**Q:** how do i put my vllm serve options in a yaml file instead of passing them on the command line  
**A:** Write the args into a YAML file using their long-form names (e.g. model, port, uvicorn-log-level), then run `vllm serve --config config.yaml`. If you also pass an arg on the command line, the CLI value wins.  
**Source:** `configuration/serve_args.md` › Configuration file  
**Evidence:** You can load CLI arguments via a [YAML](https://yaml.org/) config file.
The argument names must be the long form of those outlined [above](serve_args.md).

### q015 · config_flag
**Q:** I only want lora applied to o_proj, not every layer. what flag do i use?  
**A:** Use `--lora-target-modules`, e.g. `--lora-target-modules o_proj`. If you leave it out, LoRA goes on every supported module.  
**Source:** `features/lora.md` › Restricting LoRA to Specific Modules  
**Evidence:** The `--lora-target-modules` parameter allows you to restrict which model modules have LoRA applied at deployment time.

### q016 · config_flag
**Q:** api server is the bottleneck with a large data parallel deployment, what flag lets me run more of them?  
**A:** Use `--api-server-count` (e.g. `--api-server-count=4`); it still exposes a single HTTP endpoint/port.  
**Source:** `serving/data_parallel_deployment.md;configuration/optimization.md;design/arch_overview.md;serving/expert_parallel_deployment.md` › Internal Load Balancing  
**Evidence:** In this case, the orthogonal `--api-server-count` command line option can be used to scale this out (for example `--api-server-count=4`).

### q017 · config_flag
**Q:** using fp8 kv cache but want the sliding window attention layers kept at full precision, which flag?  
**A:** Use --kv-cache-dtype-skip-layers, e.g. `--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window`. It also accepts layer indices like `0 1 23`.  
**Source:** `features/quantization/quantized_kvcache.md` › Skipping Specific Layers from KV-Cache Quantization  
**Evidence:** The `--kv-cache-dtype-skip-layers` flag leaves the specified layers at the model's native dtype while keeping the rest of the layers under the chosen quantized dtype.

### q018 · multi_hop
**Q:** if i turn on enforce_eager to save gpu memory, what performance do i give up?  
**A:** enforce_eager turns off CUDA graph capture entirely, which frees the extra GPU memory the graphs use. The cost is slower steady-state decode, although startup gets faster.  
**Source:** `configuration/conserving_memory.md;configuration/optimization.md` › Reduce CUDA Graphs  
**Evidence:** You can disable graph capturing completely via the `enforce_eager` flag: || Skips both compilation and CUDA-graph capture for the fastest possible startup, at the cost of steady-state decode performance.

### q019 · config_flag
**Q:** whisper endpoint rejects my big audio files, is there a setting to raise the upload size limit?  
**A:** Set the `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable; it defaults to 25 MB.  
**Source:** `serving/online_serving/speech_to_text.md;usage/security.md` › API Enforced Limits  
**Evidence:** Set the maximum audio file size (in MB) that VLLM will accept, via the
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable. Default is 25 MB.

### q020 · multi_hop
**Q:** I'm persisting the compile cache in docker, how can I make sure vllm actually uses it rather than quietly recompiling?  
**A:** Mount a named volume at the cache root (`-v vllm-cache:/root/.cache/vllm`) and set `VLLM_FORCE_AOT_LOAD=1` so vLLM fails loudly instead of silently recompiling on a cache miss.  
**Source:** `deployment/docker.md;configuration/optimization.md` › Persist the compile cache across containers  
**Evidence:** named volume at that path to reuse the inductor, Triton, and AOT artifacts from || Set `VLLM_FORCE_AOT_LOAD=1` to fail loudly instead of silently recompiling when the cache misses

### q021 · unanswerable
**Q:** how much higher throughput does vllm get than TensorRT-LLM on llama 3 70B?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped whole docs folder for 'TensorRT-LLM', 'TensorRT', 'TRT-LLM', 'faster than', 'compared to (sglang|tgi|tensorrt)': only hits are FlashInfer TRT-LLM kernel/backend names, no head-to-head throughput numbers.

### q022 · how_to
**Q:** model fails to load because its config.json has no architectures field, how do i fix that?  
**A:** Tell vLLM the architecture yourself by overriding config.json with `hf_overrides`, e.g. `hf_overrides={"architectures": ["GPT2LMHeadModel"]}`.  
**Source:** `configuration/model_resolution.md`   
**Evidence:** To fix this, explicitly specify the model architecture by passing `config.json` overrides to the `hf_overrides` option.

### q023 · unanswerable
**Q:** when is batch invariance expected to come out of beta?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped the whole docs folder for 'batch invarian' together with beta/GA/general availability/release/roadmap/stable, plus 'out of beta', 'exit beta' and 'leave beta'. The only hit is the note that the feature is 'currently in beta' with a tracking-issue link. No date or timeline is given.

### q024 · how_to
**Q:** how can I send top_k through the openai python client to vllm?  
**A:** Put it in the request's `extra_body`, e.g. `extra_body={"top_k": 50}`.  
**Source:** `serving/online_serving/openai_compatible_server.md` › Completions API  
**Evidence:** You can pass these parameters to vLLM using the OpenAI client in the `extra_body` parameter of your requests, i.e. `extra_body={"top_k": 50}` for `top_k`.

### q025 · how_to
**Q:** can i run a vision-language model for text only so it doesn't reserve memory for images?  
**A:** Yes. Set that modality's limit to zero with `limit_mm_per_prompt`, e.g. `limit_mm_per_prompt={"image": 0}`, and vLLM won't accept images.  
**Source:** `configuration/conserving_memory.md` › Multi-modal input limits  
**Evidence:** You can even run a multi-modal model for text-only inference:

### q026 · how_to
**Q:** how can I stretch a qwen model's context window with yarn when serving?  
**A:** Pass `--hf-overrides` with a `rope_parameters` dict (e.g. `rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 32768`) and raise `--max-model-len`, e.g. to 131072. The old `--rope-scaling` flag no longer works.  
**Source:** `features/context_extension.md` › Usage  
**Evidence:** Run the vLLM server with the following command to extend the context length using YARN:

### q027 · factual
**Q:** how many previous tokens does the watermark use as context by default?  
**A:** context_width defaults to 4 prior tokens. Larger values make the watermark less robust to edits, and values above 16 trigger a warning.  
**Source:** `features/watermarking.md` › Configuration  
**Evidence:** and defaults to 4. Larger values make the watermark less robust to

### q028 · config_flag
**Q:** is there a way to start vllm without loading the real weights, to check if weight loading is what's slow?  
**A:** Pass --load-format dummy. It skips loading the model weights, so you can tell whether downloading or loading is the bottleneck.  
**Source:** `usage/troubleshooting.md;design/huggingface_integration.md` › Hangs loading a model from disk  
**Evidence:** you can use the `--load-format dummy` argument to skip loading the model weights

### q029 · multi_hop
**Q:** what env vars does the vllm server need to use the ipc weight transfer backend over HTTP?  
**A:** Set `VLLM_SERVER_DEV_MODE=1` so the HTTP weight transfer endpoints exist, and `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (on server and client) because IPC handles are pickled for HTTP.  
**Source:** `training/weight_transfer/README.md;training/weight_transfer/ipc.md` › API Endpoints  
**Evidence:** The HTTP weight transfer endpoints require `VLLM_SERVER_DEV_MODE=1` to be set. || When using HTTP transport, you must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on both the server and client.

### q030 · how_to
**Q:** how do i turn on sleep mode for the openai server so I can hit /sleep and /wake_up?  
**A:** Start the server with the env var VLLM_SERVER_DEV_MODE=1 and pass --enable-sleep-mode, e.g. `VLLM_SERVER_DEV_MODE=1 vllm serve MODEL --enable-sleep-mode`. Then POST to /sleep?level=1 and /wake_up.  
**Source:** `features/sleep_mode.md` › Online Serving  
**Evidence:** To enable sleep mode in a vLLM server you need to initialize it with the flag `VLLM_SERVER_DEV_MODE=1` and pass `--enable-sleep-mode` to the vLLM server.

### q031 · factual
**Q:** if I pass --max-model-len 32K is that 32000 or 32768?  
**A:** 32768. Uppercase suffixes like K are binary (1K = 1,024), while lowercase k is decimal (1k = 1,000).  
**Source:** `cli/README.md` › serve  
**Evidence:** Binary suffixes (`K`, `M`, `G`, `T`) require integers: `32K` = 32,768.

### q032 · how_to
**Q:** how do i stop vllm from sending anonymous usage stats?  
**A:** Set VLLM_NO_USAGE_STATS=1 or DO_NOT_TRACK=1, or create the file ~/.config/vllm/do_not_track. Any one of these works.  
**Source:** `usage/usage_stats.md` › Opting out  
**Evidence:** You can opt out of usage stats collection by setting the `VLLM_NO_USAGE_STATS` or `DO_NOT_TRACK` environment variable, or by creating a `~/.config/vllm/do_not_track` file

### q033 · how_to
**Q:** pdb breakpoints in vllm just throw BdbQuit, how do I debug?  
**A:** The breakpoint runs in a subprocess. Use forked-pdb (pip install fpdb, then __import__('fpdb').ForkedPdb().set_trace()), or set VLLM_ENABLE_V1_MULTIPROCESSING=0 so normal pdb works.  
**Source:** `usage/troubleshooting.md` › Breakpoints  
**Evidence:** Another option is to disable multiprocessing entirely, with the `VLLM_ENABLE_V1_MULTIPROCESSING` environment variable.

### q034 · factual
**Q:** what's the default seed in vllm v1?  
**A:** It's 0 by default in V1, which seeds every worker so runs stay consistent even with temperature > 0. You can't un-set it.  
**Source:** `usage/reproducibility.md` › Default Behavior  
**Evidence:** In V1, the `seed` parameter defaults to `0` which sets the random state for each worker

### q035 · factual
**Q:** what all2all backend does vllm use for expert parallel if I don't pick one?  
**A:** It defaults to `allgather_reducescatter`, the general-purpose backend that works with any EP+DP setup.  
**Source:** `serving/expert_parallel_deployment.md` › Backend Selection Guide  
**Evidence:** | `allgather_reducescatter` | Default backend | Standard all2all using allgather/reducescatter primitives |

### q036 · factual
**Q:** which structured outputs backend does vllm serve pick if I don't set one?  
**A:** The default backend is `auto`, which chooses a backend based on the request. You can override it with --structured-outputs-config.backend.  
**Source:** `features/structured_outputs.md` › Online Serving (OpenAI API)  
**Evidence:** `--structured-outputs-config.backend` flag to `vllm serve`. The default backend is `auto`,

### q037 · config_flag
**Q:** is there a flag to make vllm serve use the hugging face transformers implementation of a model even though vllm has its own?  
**A:** Yes, pass `--model-impl transformers` to `vllm serve` (or `model_impl="transformers"` for offline inference).  
**Source:** `models/supported_models.md` › Transformers  
**Evidence:** set `model_impl="transformers"` for [offline inference](../serving/offline_inference.md) or `--model-impl transformers` for the [online serving]

### q038 · factual
**Q:** can one vllm openai server serve multiple models on the same port?  
**A:** No, the server hosts one model at a time. Run a separate server instance per model and put a routing layer in front of them.  
**Source:** `usage/faq.md;getting_started/quickstart.md`   
**Evidence:** that is not currently supported, you can run multiple instances of the server (each serving a different model) at the same time

### q039 · how_to
**Q:** is there a way to make engine restarts faster by keeping weights in gpu memory?  
**A:** Yes: run `vllm preload --model <model>` to start one weight cache daemon per GPU, then start the engine with `--load-format ipc_cache` so it maps the cached weights over CUDA IPC instead of loading them from disk.  
**Source:** `features/preload.md;cli/README.md` › Quick start  
**Evidence:** Then start (or restart) engines with the `ipc_cache` load format:

### q040 · how_to
**Q:** vllm bench isn't working after a plain pip install, what do I need to install?  
**A:** The benchmark commands need extra dependencies, so install with pip install vllm[bench].  
**Source:** `cli/README.md` › bench  
**Evidence:** To use benchmark commands, please install with extra dependencies using `pip install vllm[bench]`.

### q041 · unanswerable
**Q:** how long does each vllm release keep getting security patches?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped whole docs folder (case-insensitive) for: backport, support window, maintained for, patch release, release cadence, release schedule, lifecycle, LTS, long-term support, end of life, EOL, security fix, security update, supported versions, months. No release support or patch window is stated; usage/security.md only links to the external SECURITY.md for reporting vulnerabilities.

### q042 · unanswerable
**Q:** what's the max number of concurrent websocket sessions the /v1/realtime endpoint supports?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped whole docs folder for 'realtime' combined with 'max|limit|concurren|session|timeout', 'max.*websocket', 'max_connections': realtime docs describe protocol/events and models only, no session concurrency limit.

### q043 · how_to
**Q:** can vllm bench serve gradually increase the request rate during a run?  
**A:** Yes, set `--ramp-up-strategy` to `linear` or `exponential` and give the start and end rates with `--ramp-up-start-rps` and `--ramp-up-end-rps`.  
**Source:** `benchmarking/cli.md` › Running With Ramp-Up Request Rate  
**Evidence:** - `--ramp-up-strategy`: The ramp-up strategy to use (`linear` or `exponential`).

### q044 · config_flag
**Q:** i want vllm to pull models from modelscope instead of hugging face  
**A:** Set the environment variable VLLM_USE_MODELSCOPE=True before you start the engine.  
**Source:** `getting_started/quickstart.md;models/supported_models.md` › Offline Batched Inference  
**Evidence:** If you would like to use models from [ModelScope](https://www.modelscope.cn), set the environment variable `VLLM_USE_MODELSCOPE` before initializing the engine.

### q045 · unanswerable
**Q:** roughly what does an A100 cost per hour if I run vllm on it in the cloud?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** grep for A100 price/pricing/per hour/hourly: only an example dstack offer table for L4 GPUs, no A100 pricing

### q046 · unanswerable
**Q:** how many dollars per million tokens does prefix caching save on cloud GPUs?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped the whole docs folder for 'per million tokens', 'dollar', 'pricing', 'cost savings', '$/', and 'cost' near prefix/cache terms. None give any monetary savings figure for prefix caching.

### q047 · factual
**Q:** what's the default memory chunk size for sleep mode on ROCm?  
**A:** It's 256MB, controlled by VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE (in MB). Lower it if you hit OOM; powers of 2 are recommended.  
**Source:** `features/sleep_mode.md` › Limitation  
**Evidence:** The default value is set at 256MB.

### q048 · factual
**Q:** what's the minimum nvidia gpu needed for batch invariance?  
**A:** You need an NVIDIA GPU with compute capability 8.0 or higher (Intel XPUs with Triton support also work).  
**Source:** `features/batch_invariance.md` › Hardware Requirements  
**Evidence:** - NVIDIA GPUs with compute capability 8.0 or higher.

### q049 · how_to
**Q:** how do i stop vllm docker containers from recompiling torch.compile every time they start?  
**A:** Mount a named volume at the vLLM cache root (default ~/.cache/vllm), e.g. `-v vllm-cache:/root/.cache/vllm`, so inductor/Triton/AOT artifacts are reused from the second container on.  
**Source:** `deployment/docker.md` › Persist the compile cache across containers  
**Evidence:** named volume at that path to reuse the inductor, Triton, and AOT artifacts from

### q050 · factual
**Q:** how often does EPLB rebalance experts by default?  
**A:** Every 3000 engine steps by default (the `step_interval` key in `--eplb-config`).  
**Source:** `serving/expert_parallel_deployment.md` › EPLB Parameters  
**Evidence:** | `step_interval` | Frequency of rebalancing (every N engine steps) | 3000 |

### q051 · factual
**Q:** which distributed runtime does vllm use by default on a single node?  
**A:** Native Python `multiprocessing` for single-node; Ray is the default for multi-node. You can override it with `--distributed-executor-backend`.  
**Source:** `serving/parallelism_scaling.md` › Single-node deployment  
**Evidence:** The default distributed runtimes are [Ray](https://github.com/ray-project/ray) for multi-node inference and native Python `multiprocessing` for single-node inference.

### q052 · multi_hop
**Q:** on intel xpu can I run batch invariant mode together with CPU KV offloading?  
**A:** Yes. Batch invariance supports Intel XPUs with Triton, using the TRITON_ATTN attention backend, and the OffloadingConnector supports CUDA, ROCm and XPU.  
**Source:** `features/batch_invariance.md;features/kv_offloading_usage.md` › Hardware Requirements  
**Evidence:** - Intel XPUs with Triton support. || The `OffloadingConnector` currently supports CUDA, ROCm, and XPU only.

### q053 · how_to
**Q:** how do i add a new lora adapter to a running vllm server without restarting it?  
**A:** Start the server with `--enable-lora`, set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, then POST `lora_name` and `lora_path` to the `/v1/load_lora_adapter` endpoint (use `/v1/unload_lora_adapter` to remove it).  
**Source:** `features/lora.md;usage/security.md` › Dynamically serving LoRA Adapters  
**Evidence:** To dynamically load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter` endpoint with the necessary

### q054 · unanswerable
**Q:** how many requests per second can the production-stack router handle before it becomes the bottleneck?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** grep router + requests per second/rps/qps/throughput under deployment/: no matches

### q055 · unanswerable
**Q:** how much does paid enterprise support for vllm cost?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped whole docs folder (case-insensitive) for: pricing, price, SLA, enterprise support, support plan, subscription, commercial support, paid support, enterprise, support contract, cost per. Only unrelated hits (dstack instance price table, benchmark SLA thresholds, a GitHub watch button, Kthena 'enterprise-grade' blurb); nothing on vLLM support plans or costs.

### q056 · unanswerable
**Q:** is vllm's json structured output faster than sglang's?  
**A:** Not covered in the vLLM docs.  
**Source:** `—`   
**Evidence:** Grepped whole docs folder for 'sglang' (only design/prefix_caching.md, which mentions SGLang only as using prefix caching); no sglang lines mention struct/json/grammar/xgrammar/guided; features/structured_outputs.md has no faster/latency/overhead/benchmark content.

### q057 · config_flag
**Q:** I want to send a trickle of tiny side requests during a benchmark to see how much the main load slows them down, which option does that?  
**A:** Use `--probe-request-rate` with a positive value; it sends single-token text-only probe requests at that rate and reports their latency separately.  
**Source:** `benchmarking/cli.md` › Probe Requests  
**Evidence:** Setting `--probe-request-rate` to a positive value sends single-token text-only
probe requests at that rate (requests per second) alongside the main workload.

### q058 · config_flag
**Q:** env var to get the same output no matter how requests get batched together  
**A:** Set `VLLM_BATCH_INVARIANT=1`.  
**Source:** `features/batch_invariance.md` › Enabling Batch Invariance  
**Evidence:** Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

### q059 · config_flag
**Q:** sweep startup just warns on unknown keys in my params json, how do i make it error out instead?  
**A:** Pass `--strict-params` to `vllm bench sweep startup` so unknown keys fail fast instead of being ignored with a warning.  
**Source:** `benchmarking/sweeps.md` › Startup Benchmark  
**Evidence:** By default, unsupported parameters in `--serve-params` or `--startup-params` are ignored with a warning.
    Use `--strict-params` to fail fast on unknown keys.

### q060 · multi_hop
**Q:** what do i put in my vllm serve config.yaml to load weights with the run:ai model streamer?  
**A:** Add `load-format: runai_streamer`. YAML keys use the long-form CLI arg names, and the Run:ai streamer is enabled with `--load-format runai_streamer` (after `pip install vllm[runai]`).  
**Source:** `configuration/serve_args.md;models/extensions/runai_model_streamer.md` › Configuration file  
**Evidence:** The argument names must be the long form of those outlined [above](serve_args.md). || To run it as an OpenAI-compatible server, add the `--load-format runai_streamer` flag:
