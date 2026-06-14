# Contributing

Thanks for helping improve `fast-ddim-tf`. This project is focused on a clear TensorFlow DDIM implementation, reproducible experiments, and math notes that match the code.

## Good First Contributions

- Improve README examples, result tables, and generated sample figures.
- Add focused tests for diffusion math, data loading, or shape handling.
- Fix items from `TODO_src_fddim_review.md`.
- Add new CIFAR-10 network configs and fill in the README result placeholders.
- Improve docs in `docs/math_note/` when the code or derivation changes.

## Development Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a syntax check before opening a pull request:

```bash
python -m py_compile src/fddim/*.py scripts/*.py
```

For a quick config-generation check:

```bash
python scripts/run_cifar10_scheduler_pred_sweep.py --dry-run
```

## Pull Request Guidelines

- Keep changes scoped to one behavior or experiment at a time.
- Include the config and command used for any reported FID number.
- Do not commit large training outputs, checkpoints, generated image sets, or local caches.
- If you change DDIM/DDPM math, update both `src/fddim/diffusion_utils.py` and `docs/math_note/` when needed.
- If you change public config keys or CLI behavior, update README examples and config comments.

## Reporting Results

When adding experiment results, include:

- network config file
- scheduler
- prediction type
- loss weighting
- checkpoint epoch
- number of generated images
- number of real reference images
- reverse steps and DDIM eta
- FID value

## Code Style

- Prefer existing code patterns and TensorFlow/Keras APIs already used in the repo.
- Keep comments short and focused on non-obvious behavior.
- Avoid unrelated refactors in experiment/result update pull requests.
