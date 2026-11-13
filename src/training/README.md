# Training entrypoints

Use `radar-train --help`, `radar-evaluate --help` and `radar-adapt --help` after installing the package. See [the training guide](../../docs/training.md) for dataset roles, feature-contract requirements and evaluation boundaries.

Training, target adaptation and reference scoring are separate operations. Supply explicit input/output paths and a single labeled source dataset; do not mix target labels into source training.
