## Summary

Describe the user-facing and analytical change.

## Data and calculation review

- [ ] Sources and as-of dates are disclosed or unchanged.
- [ ] Missing observations are not replaced with fabricated values.
- [ ] Benchmark alignment and any forward-fill behavior are explicit.
- [ ] Historical calculations do not use future observations.

## Validation

- [ ] Relevant unit/regression tests pass locally.
- [ ] `ruff check --select E,F,I,B --ignore E501 adfm_core tests` passes.
- [ ] The changed Streamlit page renders without a traceback.

## Risk and rollout

List material model, data-provider, UI, or backwards-compatibility risks.
