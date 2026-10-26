# Contributing

Thanks for helping. Bug reports, new formats and fixes are all welcome.

## Reporting a bug

Open an issue with the file that fails, if you can share it, and what meshport did with it (the error, or what looks wrong in the output). For a file you can't share, `meshport yourfile.fbx` prints what meshport reads from it, which often helps.

Security problems go through [SECURITY.md](SECURITY.md) instead, not a public issue.

## Making a change

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
```

Tests that need Blender, Unity or a folder of real files skip when those aren't there. GitHub Actions runs the tests (Python 3.10 and 3.13) and `ruff check` on every pull request.

- **Write from the format's specification**, never from GPL code (Blender's importers, for example), so meshport stays MIT.
- **New format:** one reader and one writer that go through `Scene` (see `scene.py`), plus round-trip and damaged-file tests in the style of `tests/test_damaged.py`.
- **Hostile files:** anything a reader can be fed must raise `MeshportError` and nothing else. Don't trust sizes, counts or offsets from the file, and never follow paths in a file to network locations or to files that aren't images (see `TextureFinder` and `read_image_file`).
- **Pure Python and NumPy only.** No compiled extensions.
- **Changelog:** add a dated entry to [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.

## License

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).
