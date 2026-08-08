# Architecture

NVIDIA MIG has two main instance layers:

```text
Physical GPU
└── GPU Instance (GI)
    └── Compute Instance (CI)
```

MIGOps exposes both a simple workflow and an advanced workflow.

Normal administration:

```text
status / split / create / destroy / mode
```

Advanced administration:

```text
gi / ci
```

Desired-state management sits above both:

```text
YAML
  ↓
validate
  ↓
diff
  ↓
plan
  ↓
apply
  ↓
nvidia-smi
  ↓
GPU
```

The project intentionally uses NVIDIA's native tooling as the hardware authority rather than maintaining a hard-coded GPU profile database.
