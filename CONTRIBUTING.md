# Contributing to attnroute

We welcome contributions! Here's how you can help.

## Reporting Issues

### Bug Reports

Found a bug? Generate a diagnostic report and [open an issue](https://github.com/jeranaias/attnroute/issues/new).

**Generate diagnostic report:**
```bash
attnroute diagnostic
```

This creates `attnroute_diagnostic.txt` with all the information we need.

Include in your issue:
1. **What happened** - Clear description of the bug
2. **What you expected** - What should have happened
3. **How to reproduce** - Steps to reproduce the issue
4. **Diagnostic report** - Attach `attnroute_diagnostic.txt`
5. **Logs** - Any error messages or stack traces

### Performance Issues

If attnroute is slower or less accurate than expected:

1. Run the verification benchmark:
   ```bash
   python benchmarks/verify_claims.py /path/to/your/repo
   ```

2. Include the output in your issue

3. Share details about your codebase:
   - Number of files
   - Primary language(s)
   - Approximate total lines of code

### Feature Requests

Have an idea to improve attnroute? [Open an issue](https://github.com/jeranaias/attnroute/issues/new) with:

1. **Use case** - What problem are you trying to solve?
2. **Proposed solution** - How do you envision it working?
3. **Alternatives** - Other approaches you've considered

## Contributing Code

### Setup

```bash
# Clone the repo
git clone https://github.com/jeranaias/attnroute.git
cd attnroute

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/verify_claims.py
```

### Before Submitting

1. **Run benchmarks** - Ensure performance hasn't regressed:
   ```bash
   python benchmarks/verify_claims.py
   ```

2. **Run linting** - Check code style:
   ```bash
   ruff check attnroute/
   ```

3. **Add tests** - For new features or bug fixes

4. **Update docs** - If behavior changes

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and benchmarks
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

### Code Style

- Follow existing patterns in the codebase
- Use type hints where practical
- Keep functions focused and testable
- Document non-obvious behavior

## Benchmark Contributions

We especially value contributions that improve our benchmarks:

- **New test repositories** - Diverse codebases help validate claims
- **Comparison tools** - Fair comparisons with other tools
- **Methodology improvements** - More rigorous testing approaches

## Questions?

- [Open a discussion](https://github.com/jeranaias/attnroute/discussions)
- Check existing issues for similar questions

Thank you for helping make attnroute better!
