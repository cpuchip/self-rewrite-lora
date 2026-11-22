# Lab log

Newest last. Times are UTC. A wrong entry is struck through and corrected below it, never deleted.

## 2026-11-22

- Saw the Better Stack short about the self-rewriting paper. Wrote down the idea as an adapter (see
  [method.md](method.md)) and parked it until we had time.
- Unparked the same afternoon. Split into the three steps in the [README](../README.md): serving (vllm), the edit
  (threadchip), and the model as the experimenter (after the first two report).
- Read the vLLM documentation on LoRA: adapters are chosen per request by name and can be loaded at runtime. The
  documentation says nothing either way about LoRA together with a speculative-decoding drafter, so step 1 tests it.
- The paper's code is not public and we have not read the paper itself. The method notes are from the video only.

### Step 1, serving: measured (four boots, 21:09-21:28; the first failed on the flag below)

Setup: one RTX 4090 under WSL; Qwen3.8-27B as a W4A16 AutoRound quantization, with its embeddings and output head requantized to int8, on our
vLLM 0.34 fork, started with the
same command as our production server, context 65,536 tokens; greedy decoding, thinking off, six fixed prompts, one
request at a time. The adapters were rank one, on layer 8's `linear_attn.out_proj` (a Gated DeltaNet layer) and layer
7's `self_attn.o_proj`, at zero and two scales. Directions were random: this step tests serving, not the edit.

- **Per request, by name: yes.** A request naming an adapter got it; a request naming the base model got the base, on
  the same server.
- **On the fly: yes.** Five adapters loaded with `POST /v1/load_lora_adapter` (200 each) and one unloaded (200), after
  which a request naming it got 404; no restart, and the base model's output was identical before and after.
- **Beside the speculative drafter: yes.** The drafter proposes from the base weights (vLLM's own config says drafters
  do not run LoRA adapters), and its acceptance did not drop: 3.469 tokens per step (tokens emitted per drafter round) with zero adapters, the
  same as the base, and 3.56-3.70 with the random adapters, on texts that differ. Turning the drafter off instead cost
  68% of decode speed (57.8 against 181.7 tokens per second), so it stays on.
- **Controls.** All-zero adapters gave text identical to the base on 6 of 6 prompts at the same tokens per step. With
  LoRA enabled but no adapter named, text was identical, on 6 of 6, to a server started without LoRA, and decode was 181.7 against 182.8 tokens per second
  (-0.6%). Requests with an adapter decoded about 2.3% slower, and the KV cache pool was 0.27% smaller. The random
  adapters did change the text, on 3-4 of 6 prompts, so the test could see a change when there was one.
- **One flag our checkpoint needs.** Plain `--enable-lora` fails at load on our model's int8 embedding, because vLLM's
  LoRA embedding layer reads a full-precision weight that is not there. Restricting LoRA to the two projections avoids
  it:

  ```
  --enable-lora --max-lora-rank 8 --max-loras 2 --lora-target-modules o_proj out_proj
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
  ```

  The runtime endpoints let anyone holding the server's key load files from the server's disk, so they belong only on
  a server nobody else can reach.
- **One of our own predictions was wrong.** We wrote down before the run that requests with an adapter would lose
  drafter acceptance. They did not.
- **Not checked yet:** any other card, the full 160k context, ranks above one, adapters with real directions (step 2's),
  more than one adapter in a batch, and our production server.
