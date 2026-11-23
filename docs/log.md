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

## 2026-11-23

### Step 1, reviewed (10:37-10:48)

- threadchip recomputed every step 1 number from the raw files, and all of them match. One gap remained: the evidence
  showed that an adapter changes the output, but not that the server applies exactly the specified change, including
  LoRA's `lora_alpha / r` scale. So threadchip checked it on one boot (an RTX 3090, the same image and launch script,
  with speculative decoding off because hidden-state extraction needs it), reading the hidden state just before and just after layer 8:
  - every adapter left the state before layer 8 bit-identical to the base;
  - at `a = 1` the part of layer 8's output along `d` vanished (least-squares slope +0.0011 against the base's);
  - at `a = 0.05` the slope was 0.9507, the specified `1 - a`;
  - the same change written as `r = 1, lora_alpha = 2` with `B` halved gave the same state within 3.8e-5 (relative),
    so the server applies the scale.

  Verdict: pass.

### Step 2, first pass: our own agent logs (11-22 21:10 to 11-23 04:06)

- Our records held only 4 replies from this model that ran out of tokens, among 5,255 stored messages: too few to find a
  direction. So threadchip reran 149 stored prompts from one of our agents (it reviews documents and returns a JSON
  verdict), each with the system prompt the record had sent, at temperature 0 and a 12,000-token cap, on an RTX 3090
  with our production checkpoint and launch script.
- Only 8 of the 149 ran out at 12,000. We had predicted that at least half of the long-reasoning prompts would; 6 of
  59 did. We had also predicted that half of the runs that ran out would end in a verbatim loop; 2 of 8 did, and the
  rest were still re-checking their sources or still weighing a verdict at the cap.
- Eight cases cannot make a find set, so the label moved to a 6,000-token budget (38 of 149 ran out). Under greedy
  decoding a 6,000-token run is a prefix of the 12,000-token run, so this is the label a 6,000-token request would get.
  But 30 of those 38 went on to finish by 12,000, so most "runaways" at this budget were long, healthy answers.
- **Hidden states.** vLLM 0.34's own `extract_hidden_states` with its example connector, where state `k` is the
  residual stream after `k` layers. Three checks gate every capture: state 0 equals the model's embedding rows; the
  last state, through the final norm and output head, gives the server's own top token; a repeated request gives
  identical states. "Layer 8" can mean two things, and neither the video nor the paper says which: state 9 (the output
  of 0-indexed layer 8, a Gated DeltaNet linear-attention layer) or state 8 (Hugging Face's `hidden_states[8]`, the
  output of layer 7, a full-attention layer). We captured both. One trap: vLLM returns the states in ascending layer
  order, whatever order they were requested in. A repeat of state 9 in a second boot caught the swap before anything
  was read.
- **Separation**, as held-out AUC (the chance that a randomly picked answer that ran out scores above a randomly picked
  one that finished; 0.5 is chance, 1.0 is perfect), on 72 held-out runs, 19 of which ran out:
  - prompt length alone: 0.729;
  - the registered direction (the difference of means over the first 512 reasoning tokens, state 9): 0.59;
  - the best cell, the first 128 reasoning tokens: 0.843, but unstable (directions found from the two halves had
    cosine 0.52, and 80% subsamples scored 0.72).

  No direction here beat prompt length once sampling luck was taken out.
- **Edits at `a = 1`** (twenty times the paper's 0.05) on layer 8's and layer 7's output projections, with a
  random-direction control, on 75 held-out prompts at a 6,000-token cap, all in one boot:
  - base: 19 of 75 ran out;
  - the registered layer-8 edits: 19 and 20;
  - layer 7's attention output: 11 with the registered direction and 12 with an exploratory one (the first 128
    tokens), the right way but neither significant (sign test p 0.19 and 0.21), and not independent, since both ran
    on the same prompts;
  - layer 7's MLP output (registered): 18;
  - the random control: 25, a move of 6 where we had predicted fewer than 3;
  - quality held on every arm: perplexity within 0.2%, GSM8K 0.945-0.960 against the base's 0.955 (200 problems,
    thinking off).
- **What that pass taught us.** Rebooting the same base reproduced only 52 of 75 labels, and every adapter changed the
  text or length of 65-74 of the 75 answers. With long reasoning, any perturbation re-rolls the whole path, a reboot
  included, so one draw per prompt cannot resolve an effect of a few prompts. Therefore we moved to the paper's own
  protocol, which shares sampling seeds between edited and unedited runs over many problems.

### The paper, read (about 03:15 to 04:24)

- vllm and threadchip read the paper through its HTML version (arXiv 2611.21885, v2). It has not yet been checked
  against the PDF. What we use from it:
  - Qwen3.5 at 2B, 4B, 9B and 27B, on math: 3,000 practice problems from DeepScaleR, with overlaps against the common
    math benchmarks removed, and 1,000 more held out.
  - Thinking on, temperature 0.6, top-k 20, top-p 0.95, a seed per problem shared between edited and unedited runs,
    16,384 new tokens, graded by Math-Verify on the last `\boxed{}` after thinking. An answer still thinking at the
    budget scores 0. An answer is "looping" when it is cut at the budget and its last 6,000 characters compress (zlib)
    to under 15% of their size.
  - The 27B's first edit removed 5% (`a = 0.05`) of one direction from both of layer 8's output matrices, a change of
    0.6% and 0.1% of their norms. The direction is the normalized difference between the mean layer-8 hidden states of
    20 unfinished and 20 finished answers; which tokens were averaged is not stated. On 490 held-out problems (eight
    answers each) it went from 57.2% to 58.3% (p 0.038).
  - **A correction to the video's account in the README:** the +6.6 points came from a different edit, which raised the
    time-step bias (`dt_bias`) of every linear-attention layer by 0.5.

### Step 2 on the paper's protocol (04:57-08:23)

- **Setup.** DeepScaleR (MIT license) with the paper's exclusions and our own seeded 3,000 / 1,000 split; the paper's
  prompt and sampling; seed = problem id; 16,384 tokens; Math-Verify. Served on our RTX 4090 with our production
  server's command and its int4 checkpoint (4-bit weights), drafter on. Below, "open" means still thinking at 16,384
  tokens.
- **Baseline.** Practice 200: 126 correct, 39 open, 1 looping. Held-out 200: 120 correct, 34 open, 0 looping. The open
  answers do not loop (the median compression of their last 6,000 characters is 41%): they keep reasoning without
  committing to an answer.
- **The direction is real on this task.** All 200 practice answers were captured in full, at states 8 and 9. The mean
  over all reasoning tokens at state 9 separates open from finished at AUC 0.899, with the direction found on one half
  of the problems and scored on the other (0.890 the other way; cosine 0.92 between the two halves' directions).
  Prompt length alone gives 0.618, and the paper's size (20 against 20) finds the same direction (cosine 0.92-0.94).
  Two cautions:
  - the score may partly encode position, since among finished answers it correlates +0.63 with reasoning length; the
    first 512 reasoning tokens alone, where both groups average the same positions, still separate at 0.767;
  - layer 8 itself writes only 2-10% of the gap, so removing 5% of the direction from layer 8 should do little.
- **The adapters**, each `W <- W - a u u^T W` written as a rank-one LoRA with `W` dequantized: the paper's edit
  (`a = 0.05` on both layer-8 output matrices; 0.51% and 0.145% of their norms, against the paper's 0.6% and 0.1%), the
  same edit at `a = 1`, two controls at `a = 1`, and layer 7's attention output at `a = 1`. The controls are a random
  direction and layer 8's second output singular vector (cosine 0.08 with `u`; the top one lines up with `u`, cosine
  0.71). It is the unaligned direction closest in edit size to `u`'s: on the linear-attention output projection,
  `|v^T W|` is 5.55 against `u`'s 9.15, while a random direction gives 1.25.
- **Result, held-out 200, each problem paired with the base's answer:**

  | arm | correct | open | open to finished / finished to open | sign test p (open) |
  |---|---|---|---|---|
  | base | 120 | 34 | | |
  | the paper's edit, a = 0.05 | 123 | 34 | 5 / 5 | 1.0 |
  | the same, a = 1 | 120 | 36 | 4 / 6 | 0.75 |
  | random direction, a = 1 | 124 | 29 | 7 / 2 | 0.18 |
  | second singular vector, a = 1 | 120 | 36 | 3 / 5 | 0.73 |

  Layer 7's arm was inside the noise too. Every arm changed the text of 198-200 of the 200 answers, so each acts like
  a reseed, and a base rerun with the same seeds changed 199 of 200 and moved open answers 4 and 3. (Before that
  rerun finished, threadchip pointed out that a same-seed rerun might measure determinism instead of reseed noise, so
  the rule for reading it, text changed first, was written down before its answers existed.) **The paper's direction
  edit does not reproduce on our setup at n = 200.**
- **The paper's +6.6 edit** (all 48 linear-attention layers' `dt_bias` raised by 0.5, applied by a load-time hook that
  changes those 48 tensors and nothing else) went the other way on our model: open 34 to 44 (12 finished to open, 2
  the reverse, sign test p 0.013), correct 120 to 114 (p 0.18), looping 0 to 4, median finished length 1,930 to 3,170
  tokens. vllm's hypothesis, untested: the paper's model looped at baseline and ours does not, and a larger time step
  (faster forgetting in those layers) trades looping for commitment.

### What the model does when it does not commit (threadchip, 04:43-06:06)

- **Readouts.** Community Jacobian lenses (a published way to read what a mid-layer state is disposed to say),
  including three fitted on Qwen3.8-27B itself, did not decode the states of our served int4 model: at layer 8 the
  model's own top token had a median rank of 14,499-66,216 under them, against 237 under the plain logit lens (one
  Qwen3.8 lens wins near the top: 13 against 20 at layer 56). Whether quantization or the lenses cause that is open.
  A tuned-lens-style readout fitted on our own captures (for each layer, a least-squares map from that layer's state
  to the final state, then the model's own output head) does read it: on held-out texts the model's top token has
  median rank 1 at layer 41, against the logit lens's 237 at the same layer.
- ~~**The commit is abrupt.** Under that readout, `</think>` (the token that ends thinking) sits around rank 20-40 at
  best, then jumps to first at the decision, and open answers never take the step.~~ That readout sees only the next
  token. Corrected at 06:06 by a probe:
- **The commit probe.** A logistic probe on layer 41's output (state 42), labelled "`</think>` comes within the next 16
  tokens", fitted on 40 finished practice answers, with its threshold at 1% false positives. On 20 other finished
  answers it finds those positions at AUC 0.947 and fires 8 or more tokens early in 18 of 20. Its firing rate, per
  1,000 tokens:
  - finished answers, more than 1,024 tokens before they stop: 7.6; 256-1,024 before: 31; 64-256 before: 77; 16-64
    before: 239;
  - open answers, over all 16,384 tokens: 0.78.

  So a finished answer's decision to stop builds over hundreds of tokens, and an open answer, on average, never starts
  building it. ~~A change meant to help has to create that decision; nothing in these states shows one being held
  back.~~ Refined at 10:49 and 11:55 (below): a minority of open answers fire above that floor, and those are the ones
  the commit adapter converts.

### A trained adapter for commitment (vllm, 05:19-11:27)

- Michael asked for an adapter that makes the model a little more committed, so vllm trained one by self-distillation
  (training the model on its own best answers), on a rented RTX PRO 6000 (96 GB):
  - 800 practice problems used nowhere else, 4 samples each, the paper's prompt and sampling, 8,192 new tokens, on the
    full-precision (bf16) model;
  - for each problem, the shortest correct finished sample of at most 6,144 tokens: 494 kept;
  - LoRA rank 16, `lora_alpha` 32, on every layer's attention or linear-attention output projection and MLP down
    projection; loss on the answer tokens only; AdamW at 1e-4, cosine schedule with 5% warmup, 2 epochs, 8 examples
    per update; 26 minutes of training;
  - served per request on our int4 server (`--max-lora-rank 16`). The teacher was bf16 and the student is served on
    int4, a mismatch written down before the run.
- **A review before any data existed.** counsel, a seat that reviews designs, pointed out that this selection would
  mostly teach the model to be shorter on problems it already solves: no answer that finishes after 8,192 tokens can
  enter the set, and on our int4 model those late answers are right 34 times in 51. The count confirmed it: 449 of the
  494 kept examples (91%) came from problems where all four samples finished, and problems with three of four
  unfinished contributed 4.
- **The first 200 held-out problems**, with the base's seeds:

  | | correct | open | shorter where both finish | looping |
  |---|---|---|---|---|
  | int4 base (our production server) | 120 | 34 | | 0 |
  | int4 base rerun (the noise floor) | 123 | 33 | 81 of 163, p 0.94 | 0 |
  | int4 + adapter | 120 | 30 | 108 of 164, p 6e-05 | 3 |
  | bf16 base (rental) | 122 | 28 | | 0 |
  | bf16 + adapter | 126 | 28 | 101 of 168, p 0.008 | 1 |

- ~~First reading: the adapter makes answers about 10% shorter and does not make the model more committed. On int4 it
  moved 6 open answers to finished and 2 the other way, inside the rerun's 4 and 3.~~ That was right about this 200 and
  wrong about the adapter. vllm wrote it before the registered confirmation below had run, and the headline should
  have waited for it.
- **The confirmation**, registered at 10:12 before any answer existed: the untouched held-out problems 200-399, int4,
  the base's seeds, one boot.

  | int4, base's seeds | correct | open | open to finished / finished to open | shorter where both finish | looping |
  |---|---|---|---|---|---|
  | 200-399, base | 129 | 36 | | | 0 |
  | 200-399, + adapter | 134 | 26 | 11 / 1, p 0.006 | 99 of 163, p 0.0044 | 0 |
  | all 400, base | 249 | 70 | | | 0 |
  | all 400, + adapter | 254 | 56 | 17 / 3, p 0.003 | 207 of 327, p 9e-07 | 3 |

  So on our int4 server the adapter makes the model a little more committed. Answers still thinking at 16,384 tokens
  fall from 17.5% to 14.0% of the 400 problems, and 12 of the 17 conversions are correct. Finished answers are about
  9% shorter (median paired ratio 0.910), and accuracy holds (249 to 254, p 0.44). Every prediction registered for
  this run held except one: we expected more loops on the new 200, and there were none.
- **What keeps it qualified.**
  - On bf16 the open count did not move on the first 200 (28 to 28). Whether the gain is specific to int4 needs the
    bf16 pair on 200-399.
  - The adapter adds a failure the base never shows: a short fragment repeated to the budget ("1+? 1+? ..."), whose
    last 6,000 characters compress to under 1% of their size. It happened 3 times in 400 on int4, all on problems the
    base left open while still reasoning, and once in 200 on bf16, on a problem every other run finished. The base
    looped 0 times in every run. The `?` is in the model's own output; which token it is has not yet been checked
    against the tokenizer.
- **Things that bit us**, for anyone repeating this:
  - The first rented card was listed by `nvidia-smi`, but CUDA could not initialize it. Setup now makes a real CUDA
    allocation before it reports ready. That card cost 4 minutes and about $0.09; both rentals together came to about
    $7.08.
  - In transformers, the Qwen3.5 decoder checks `self.training` before it uses gradient checkpointing, so a model left
    in eval mode never checkpoints (29.45 GiB more at 1,024 tokens). Call `model.train()`.
  - vLLM falls back to spawning its worker processes when CUDA is already up in the parent, and spawning re-imports
    the script, which built a second engine. Guard the entry point with `if __name__ == "__main__":`.

### Why it helps where it helps (threadchip, 10:22-11:55)

- Each answer was read with its own model's states, with the commit probe unchanged. (The adapter applies inside the
  extraction path: an adapter on layer 8 alone left state 8 identical and changed state 9.)
- **The first 200** (T80 note 25). On answers both models finish, the adapter's run-up to stopping has the same shape
  and arrives sooner, so the shortening comes from reasoning cut before the run-up begins. The 6 open answers it
  converted had already been firing under the base at 3.29 per 1,000 tokens, against 0.72 for the open answers it left
  open (AUC 0.908, exact one-sided permutation p 0.0007). The ratio was registered (at least 2x; it came out 4.6x).
  The AUC test was exploratory, chosen after seeing the data, on n = 6.
- So threadchip registered a test on the fresh 200-399 before those files existed on its side (11:29), with a
  falsifier: an AUC under 0.60 would strike the reading. Its data line took rows by position, which was wrong, since
  the files are in completion order. vllm caught it, and it was corrected at 11:30 (T80 note 26b) before the capture,
  with the predictions unchanged. Scored at 11:55 (T80 note 27):
  - the base-side rate separates the 11 converted answers from the 25 that stayed open at AUC 0.856, exact one-sided
    p 0.00019;
  - the rule fixed from the first 200 (more than 1.0 firing per 1,000 tokens) flags 13, and 9 of them convert:
    precision 0.69 against a base rate of 0.31, recall 9 of 11;
  - the answers that stay open stay at the floor under the adapter (0.56 per 1,000).

  All five predictions held.
- **Reading.** The adapter rescues open answers in which the decision to stop was already forming, and creates none
  where none was forming. The probe, run on the base's own states, says in advance which open answers it will rescue,
  at about twice the chance rate. Limits: int4 only; one draw per arm; the 1.0 threshold crosses boots, and on the one
  answer repeated across boots, the probe's firing agreed on 90 of its 92 reasoning states (0.978; 92 of 92 in the
  first adapter boot); 8 of the 11 conversions were correct.

### Where the plan stands

- Step 2 as the paper did it, a direction removed from one layer, did not reproduce on our setup (n = 200 paired
  problems). A trained adapter, which is a different method, moved commitment a little, and the probe shows which
  answers it moves.
- Next, as vllm and threadchip wrote them: the bf16 pair on 200-399 (vllm); bf16 states, for the probe and for whether
  quantization is what breaks the lenses (threadchip, which has no bf16 weights on its box yet); the loop tokens
  decoded with the real tokenizer;
  a second adapter built for the hard tail (counsel's design: keep correct answers that finish after 8,192 tokens,
  sample to 16,384 on hard problems, and weight the problems the model leaves open so they dominate the set), using the
  probe to find examples where no decision forms.
- Step 3, the model as the experimenter, has not started.
