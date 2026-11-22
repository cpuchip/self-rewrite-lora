# self-rewrite-lora

An attempt, done in the open, to reproduce "a language model edits its own weights to fix a failure" on an open
27B model and one consumer GPU, with the edit written as a LoRA adapter so the base model's weights never change.

**Status: in progress.** Nothing here is reproduced yet. This repo is our lab notebook: the plan, the method, the
tests we commit to before running them, and the results as they come in, good or bad. The log is
[docs/log.md](docs/log.md).

## Where the idea came from

A short video from Better Stack, ["AI Just Made Itself Smarter By Changing Its Own Weights!"](https://www.youtube.com/watch?v=VyzoltTI_NA)
(2026-11-21), describes a paper titled *Language Models Can Rewrite Their Own Weights*. As the video tells it, a Qwen
27B model got a small Python sandbox over its own weights, with no gradients, no optimizer and no internet. Asked
to work out why it got some practice math problems wrong, it looked at answers that ran out of tokens before
finishing, found a direction in its hidden states at layer 8 that separated those from answers that finished, and
weakened that direction with a rank-one update. The video reports +6.6 points on 490 problems it had not seen, and
says the authors had not published code.

We have not read the paper itself yet, only that video and secondary coverage, so every claim about it above is the
video's. When we can read the paper, this section gets corrected against it.

## Why we care

We run a Qwen 27B model at home for our own agents, and it has a failure we have seen for months: on some requests
it spends its whole token budget thinking and returns an empty or unfinished answer. That is the same failure the
paper's model went after in itself, on a model the same size.

## The approach

The edit the video describes is rank one. Removing part of a direction `d` from one layer's output is

```
W' = (I - a d d^T) W  =  W + B A,   with  B = -a d  and  A = d^T W
```

and `B A` is exactly the shape of a rank-one LoRA adapter. Written as an adapter:

- the base weights on disk never change, and unloading the adapter undoes the edit;
- vLLM can apply an adapter per request: a request names the adapter where it would name the model, so the same
  server answers with and without the edit, side by side;
- vLLM can load and unload adapters while it runs, so an edit can be tried without a restart.

The details, and what is still unchecked, are in [docs/method.md](docs/method.md).

## The plan

1. **Serving.** Show that our setup (an int4 build of the model, meaning 4-bit weights, on a fork of vLLM 0.34, with a
   speculative-decoding drafter, a small model that proposes tokens for the big one to check) applies a rank-one adapter per request and loads it at runtime; that an all-zero adapter changes nothing;
   and what it costs in speed.
2. **The edit.** From our own logs, collect requests that ran out of tokens and matched ones that finished; find the
   direction that separates them; write the edit as an adapter; measure how many held-out requests still run out, and
   check that quality on an ordinary task does not drop.
3. **The model as the experimenter.** Give the model a walled sandbox where it runs that loop itself (look at its own
   activations, propose an edit, load it, score it, keep or drop it), which is the part the paper is actually about.

Each step's tests are written down before it runs, in [docs/log.md](docs/log.md).

## How this is being made

Michael Stufflebeam with a crew of Claude Code sessions, each a long-running agent we call a seat: **vllm** keeps our model server and does
step 1, **threadchip** measures on constrained hardware and does step 2, and **workspace-basecamp** coordinates. Work is
tracked on our own board (tickets T76, T79 and T80). Mistakes stay in the log, struck through, rather than being
cleaned out.

## License

MIT. See [LICENSE](LICENSE).
