# Publishing the FinSignals SDK to PyPI

Follow these steps once. After that, publishing a new version is just bumping
the version number and running the last two commands.

---

## 1. Prerequisites

Install the build and upload tools:

```bash
pip install build twine
```

---

## 2. Create your PyPI account

Go to https://pypi.org/account/register/ and create a free account.

Then go to https://pypi.org/manage/account/token/ and create an API token.
- Scope: "Entire account" for the first upload; lock it to the `finsignals`
  project after the first successful publish.
- Copy the token — you only see it once.

---

## 3. Store your credentials

Create the file `~/.pypirc`:

```ini
[pypi]
  username = __token__
  password = pypi-YOUR_TOKEN_HERE
```

Replace `pypi-YOUR_TOKEN_HERE` with the token you copied.

Set permissions so only you can read it:

```bash
chmod 600 ~/.pypirc
```

---

## 4. Build the package

From the `finsignals-python/` directory:

```bash
python -m build
```

This creates two files in `dist/`:
- `finsignals-0.1.0.tar.gz`  (source distribution)
- `finsignals-0.1.0-py3-none-any.whl`  (wheel)

---

## 5. Check the package before uploading

```bash
twine check dist/*
```

Should output "PASSED" for both files. If not, fix the errors before
uploading — you cannot overwrite an already-published version on PyPI.

---

## 6. Upload to PyPI

```bash
twine upload dist/*
```

If your `~/.pypirc` is set up correctly this completes without prompting for
credentials. You'll see a URL like:

```
View at: https://pypi.org/project/finsignals/0.1.0/
```

Within a few minutes `pip install finsignals` works for everyone.

---

## 7. Verify the install

In a fresh virtual environment:

```bash
python -m venv /tmp/test-env
source /tmp/test-env/bin/activate
pip install finsignals
python -c "import finsignals; print(finsignals.__version__)"
```

Should print `0.1.0`.

---

## Publishing a new version

1. Update the version number in **two places**:
   - `finsignals/__init__.py` → `__version__ = "0.1.1"`
   - `pyproject.toml` → `version = "0.1.1"`

2. Delete the old dist files:
   ```bash
   rm -rf dist/
   ```

3. Build and upload:
   ```bash
   python -m build
   twine upload dist/*
   ```

That's it. PyPI does not allow re-uploading the same version number, so
always increment before building.

---

## Testing on TestPyPI first (optional but recommended)

TestPyPI is a separate instance where you can test the upload flow without
affecting the real index.

1. Create an account at https://test.pypi.org/account/register/
2. Create a token at https://test.pypi.org/manage/account/token/
3. Add to `~/.pypirc`:

```ini
[testpypi]
  username = __token__
  password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

4. Upload to TestPyPI:

```bash
twine upload --repository testpypi dist/*
```

5. Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ finsignals
```

---

## Setting up GitHub releases (optional)

Once you have a GitHub repo, you can automate publishing with a GitHub Action.
Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

Add `PYPI_API_TOKEN` as a secret in your GitHub repo settings.

After this, every time you create a GitHub Release, the package automatically
publishes to PyPI. No manual steps needed.
```
