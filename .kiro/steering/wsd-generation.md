---
inclusion: manual
---

# WSD Example Generation Coordinator

You are coordinating batch generation of Chinese word sense disambiguation training examples.

## Workflow

1. Run: `python ml/scripts/wsd_training/llm_coordinator.py 10`
2. This prints status, collects any missing rows, and returns up to 10 batch files to process
3. If output ends with "DONE", all batches are complete — stop
4. For the batch files listed after `---`, spawn a subagent to process each one
5. After all subagents complete, repeat from step 1. (DO NOT try to read each of the created batch files - it will crash your context!)

## Subagent Task

Tell each subagent exactly:

```
For the WSD batch file {batch_path1}:

1. Read ml/scripts/wsd_training/INSTRUCTIONS.md — it contains format specs, rules, and examples
2. Read the input TSV file
3. Generate sentences following the instructions exactly
4. Write output to ml/data/wsd_llm/llm_results/ with the same filename as input

Do NOT verify or validate your output. The coordinator agent's script handles validation automatically.
Just generate the sentences and write the file.
```

## Parallelism

Spawn 10 parallel subagents with one of the returned batch files each, for max throughput.

## Important

- Subagents must read INSTRUCTIONS.md themselves — do not summarize it for them
- no validation needed — llm_coordinator.py automatically checks formatting, discards broken lines, and creates "missing" batches for any gaps
- DO NOT read the created result files after the subagents finish working to check them for quality/completeness. It will crash your context window. Remember that llm_coordinator.py will take care of all of that!
- Keep subagent prompts minimal — they read the instructions file for details
