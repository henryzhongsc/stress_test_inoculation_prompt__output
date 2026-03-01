# GRPO Training Pipeline

GRPO (Group Relative Policy Optimization) training loop for MBPP code verification (RLVR).

---

## Overlong Handling

When a model generates a response that exceeds `max_tokens`, the output is truncated. Setting `filter_truncated: true` excludes these from the training loss.

### Config Schema

Add to `training` section of pipeline config:

```json
"training": {
    "overlong_handling": {
        "filter_truncated": false
    }
}
```

---

### `filter_truncated`

**What it does**: Completely excludes truncated samples from the training loss.

**How it works**:
- If `len(sampled_tokens) >= max_tokens`, the sample is marked as truncated
- Truncated samples are still used for reward/advantage computation within the group
- But they are skipped when creating training datums (not included in loss)

**When to use**: When you want a clean signal — only train on samples that the model completed naturally (ended with stop token like `<|im_end|>`).

**Trade-off**: You lose training data. If many samples are truncated, you may have very few datums per batch.

---

### Metrics

The following metrics are logged per batch:

| Metric | Description |
|--------|-------------|
| `truncated/count` | Number of truncated samples in the batch |
| `truncated/rate` | Fraction of samples that were truncated |

Raw results include per-sample field:
- `response_truncated`: Boolean indicating if sample hit max_tokens
