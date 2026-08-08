# Real Hardware Validation

Before tagging the first release, test MIGOps on at least one real MIG-capable Linux GPU host.

Read-only checks:

```bash
migops status
migops profiles --gpu 0
migops users --gpu 0
migops split --gpu 0 --instances 2
```

Desired-state checks:

```bash
migops validate config.yaml
migops diff config.yaml
migops plan config.yaml
migops apply config.yaml --dry-run
```

During an approved maintenance window, test one real lifecycle sequence:

```bash
sudo migops snapshot --gpu 0
sudo migops create --gpu 0 --profile PROFILE --dry-run
sudo migops destroy --gpu 0 --all --dry-run
```

Only execute destructive commands after confirming there are no important workloads and the GPU is safe to reconfigure.

Record the GPU model, driver version, and sanitized command output for any parser issue.
