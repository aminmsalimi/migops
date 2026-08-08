# Desired-State Configuration

MIGOps configuration format version 1 uses YAML:

```yaml
version: 1

gpus:
  - gpu: "0"
    mig_enabled: true

    instances:
      - profile: "3g.40gb"
        count: 1

      - profile: "2g.20gb"
        count: 1
```

`gpu` may identify the target by supported selector such as GPU index, UUID, or PCI bus ID.

Profiles may be specified by NVIDIA profile name or profile ID. MIGOps canonicalizes IDs to the driver-reported profile name before diff and apply.

Use:

```bash
migops validate config.yaml
migops diff config.yaml
migops plan config.yaml
```

before applying:

```bash
migops apply config.yaml --dry-run
sudo migops apply config.yaml
```
