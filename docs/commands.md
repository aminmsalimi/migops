# Commands

## Inspection

```bash
migops doctor
migops status
migops profiles [--gpu GPU] [--json]
migops users [--gpu GPU] [--json]
```

## Smart split

```bash
migops split --gpu 0 --instances 4
migops split --gpu 0 --instances 4 --apply --dry-run
migops split --gpu 0 --instances 4 --apply --yes
```

## Easy lifecycle

```bash
migops mode status --gpu 0
migops mode enable --gpu 0
migops mode disable --gpu 0

migops create --gpu 0 --profile 3g.40gb --count 1
migops destroy --gpu 0 --gi 2
migops destroy --gpu 0 --all
```

## Desired state

```bash
migops validate config.yaml
migops diff config.yaml
migops plan config.yaml
migops snapshot
migops apply config.yaml --dry-run
migops apply config.yaml --yes
migops restore snapshot.yaml --dry-run
migops restore snapshot.yaml --yes
```

## Advanced operations

```bash
migops gi --help
migops ci --help
```
