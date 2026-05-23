# Multi-model benchmark — bench_001

- Prompt: v_2
- Fixtures: 10 (sample_* under .trellis/tasks/.../fixtures)
- Verify-loop: OFF
- Hard budget cap: $4.00
- Total spend: $0.652

| model | provider | sel_acc | n_correct | mean_iou | iou_pass_50 | mean_ssim | cost_usd | mean_latency_s | mean_completion_tokens | mean_reasoning_tokens | json_retries | aborted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gpt-5.5` | openai | 29% | 2/7 | 0.000 | 0% | 0.403 | $0.6519 | 13.2 | 491 | 0 | 0 | no |
