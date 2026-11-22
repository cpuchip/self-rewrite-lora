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
