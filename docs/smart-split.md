# Smart Split

Smart Split converts a simple request such as:

```bash
migops split gpu 0 4
```

into a valid equal-profile MIG recommendation.

MIGOps detects the physical GPU memory and asks the installed NVIDIA driver which MIG profiles are supported. It then chooses the identical profile that most closely matches the requested equal-memory target.

MIGOps does not invent arbitrary MIG memory sizes.

A recommendation is read-only by default. To preview execution:

```bash
migops split gpu 0 4 --apply --dry-run
```

To execute deliberately:

```bash
sudo migops split gpu 0 4 --apply --yes
```

Poor-fit recommendations are rejected instead of being applied automatically.
