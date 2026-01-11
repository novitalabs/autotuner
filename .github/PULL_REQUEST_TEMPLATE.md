## Summary
- Fix quantization parameter overrides from task parameters (issue #2).

## Changes
- Promote quantization fields from `parameters` into `__quant__*` for grid expansion.
- Add tests to verify user-specified kv-cache dtype overrides defaults.

## Testing
- [ ] `python tests/test_quant_param_override.py`

## Checklist
- [ ] I have added/updated tests where needed.
- [ ] I have run the relevant tests.
- [ ] I have updated documentation if required.

## Notes
- Related issue: #2
