"""Test version consistency between __init__.py and pyproject.toml."""


def test_version_sync():
    """Ensure __init__.py and pyproject.toml versions match."""
    import tomllib
    from pathlib import Path

    import attnroute

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert attnroute.__version__ == data["project"]["version"], (
            f"Version mismatch: __init__.py={attnroute.__version__} "
            f"pyproject.toml={data['project']['version']}"
        )
