# Commands

MIGOps uses a readable positional syntax: `gpu <GPU>`.

## Inspection

```bash
migops status
migops profiles
migops profiles gpu 0
migops users
migops users gpu 0
```

## MIG mode

```bash
sudo migops enable gpu 0
sudo migops disable gpu 0

sudo migops enable gpu 0 --dry-run
sudo migops disable gpu 0 --dry-run
```

## Recommend a split

`recommend` is always read-only. It never changes the GPU.

```bash
migops recommend gpu 0 2
migops recommend gpu 0 4
```

## Actually split a GPU

`split` means perform the split.

```bash
sudo migops split gpu 0 2 --dry-run
sudo migops split gpu 0 2
sudo migops split gpu 0 2 --yes
```

A real split asks for confirmation unless `--yes` is supplied.

## Easy lifecycle

```bash
sudo migops create gpu 0 3g.40gb
sudo migops create gpu 0 3g.40gb --count 2

sudo migops destroy gpu 0 --gi 2
sudo migops destroy gpu 0 --all
```

## Desired state

```bash
migops validate config.yaml
migops diff config.yaml
migops plan config.yaml

migops snapshot
migops snapshot gpu 0
migops snapshot gpu 0 --output before-maintenance.yaml

sudo migops apply config.yaml --dry-run
sudo migops apply config.yaml
sudo migops apply config.yaml --yes

sudo migops restore snapshot.yaml --dry-run
sudo migops restore snapshot.yaml
sudo migops restore snapshot.yaml --yes
```

## Advanced GPU Instance operations

```bash
migops gi list gpu 0
sudo migops gi create gpu 0 3g.40gb
sudo migops gi create gpu 0 3g.40gb --with-ci
sudo migops gi delete gpu 0 1
sudo migops gi delete gpu 0 --all
```

## Advanced Compute Instance operations

```bash
migops ci list gpu 0
migops ci list gpu 0 gi 1

sudo migops ci create gpu 0 gi 1 3g.40gb
sudo migops ci delete gpu 0 gi 1 ci 0
sudo migops ci delete gpu 0 gi 1 --all
sudo migops ci delete gpu 0 --all
```

## Common flags

- `--dry-run` previews a change.
- `--yes` skips interactive confirmation.
- `--force` bypasses MIGOps workload protection, but cannot bypass NVIDIA driver restrictions.
- `--json` is available on supported read-only commands.
