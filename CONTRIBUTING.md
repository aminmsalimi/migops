# Contributing

Contributions are welcome.

For code changes:

1. create a focused branch
2. keep changes small and scoped
3. add or update tests
4. run the full test suite
5. open a pull request describing the behavior change

Run tests with:

```bash
python -m unittest discover -s tests -v
```

For NVIDIA parser changes, include sanitized sample output and the GPU/driver generation when possible.

Do not include secrets, private hostnames, user data, or sensitive infrastructure information in issues or fixtures.
