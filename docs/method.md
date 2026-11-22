# Method

What we plan to do and why each piece is shaped the way it is. Anything not yet measured on our own setup is marked
**unchecked**.

## The edit as a rank-one adapter

Take one layer whose output is added into the model's residual stream, the running sum that every layer reads from
and adds to (an attention output projection or the MLP's down projection), with weight `W` of shape `out x in`. Let `d` be a unit vector in that output space, the direction we want
to weaken, and `a` how much of it to remove (`a = 1` removes it entirely). Then

```
W' = (I - a d d^T) W = W - a d (d^T W)
```

The change `-a d (d^T W)` is an outer product of a column (`-a d`, `out x 1`) and a row (`d^T W`, `1 x in`), which is
rank one. A LoRA adapter adds `B A` to a weight, with `B` of shape `out x r` and `A` of shape `r x in`. With `r = 1`,
`B = -a d` and `A = d^T W`, the adapter is exactly this edit. (In the usual PEFT format the added term is scaled by
`lora_alpha / r`; set `lora_alpha = r = 1`, or fold the scale into `B`.)

Two things follow. The base weights are never written, so the edit is undone by not loading the adapter. And `W`
has to be read densely to compute `A`; our served model is int4, so `W` comes either from the original
full-precision weights or from dequantizing that one layer. **Unchecked:** how much the int4 rounding of `W` matters
to the edit, compared with computing `A` from full-precision weights.

## Finding the direction

The video describes the simplest version: 20 answers that finished and 20 that were cut off by the token limit,
hidden states compared at layer 8, and the direction that separates the two groups. Our first pass:

- **Cases** from our own logs only, each with where it came from: requests that ended because they hit the token
  limit with an empty or unfinished answer, and matched requests that finished.
- **Hidden states** recorded by replaying each request with its recorded answer appended (teacher forcing), so no
  new text is generated and the states are the ones the model actually had.
- **The direction** as the difference of the two groups' mean hidden states, normalized, at several layers, with how
  well it separates cases held out from finding it.

**Not stated in the video, so we will try both and report which:** which tokens' hidden states were compared (the
last prompt token, the generated thinking, or the whole answer), and which projection the update went on.

## Serving it per request

From the vLLM documentation (read 2026-11-22, not yet on our fork): a request selects an adapter by putting the
adapter's name in the `model` field; with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` the server exposes
`/v1/load_lora_adapter` and `/v1/unload_lora_adapter`, which the documentation says belong only in a trusted,
isolated environment. Our server also runs a speculative-decoding drafter. **Unchecked:** how an adapter on the main
model affects the drafter's acceptance rate and the speed.

## What would count as working

Written before any run, and each with a control that could fail:

- An all-zero adapter (`B = 0`) gives output identical to the base model under greedy decoding.
- On requests held out from finding the direction, the share that run out of tokens drops with the adapter loaded.
- On an ordinary task unrelated to the failure, the score with the adapter is not lower than without it.
- A random direction of the same size, used in place of `d`, does not produce the same drop. If it does, the drop
  came from perturbing the layer, not from the direction.
