"""Every script's --help must render.

argparse runs its help strings through `%` formatting, so a literal percent
sign has to be written `%%`. An unescaped one raises only when --help is
called, which is exactly when someone is trying to find out how to use the
script. Two have shipped broken already -- `compare_windows` ("~99% of") and
`test_delta_transform` ("~1.3% of") -- so this closes the class rather than
the instances.

Import-and-inspect rather than subprocess: a subprocess per script is slow and
would also catch unrelated import failures.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = sorted(p for p in (Path(__file__).parent.parent / "scripts").glob("*.py")
                 if not p.name.startswith("_"))
assert SCRIPTS, "no scripts found; has the layout changed?"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_help_text_formats(path, monkeypatch):
    """Build each parser and force argparse to format its help."""
    import argparse

    seen = []

    real_init = argparse.ArgumentParser.__init__

    def capture(self, *a, **kw):
        real_init(self, *a, **kw)
        seen.append(self)

    monkeypatch.setattr(argparse.ArgumentParser, "__init__", capture)
    # Stop the script at parse time; we only want the parser built.
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, *a, **kw: (_ for _ in ()).throw(SystemExit(0)))

    spec = importlib.util.spec_from_file_location(f"_help_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    if hasattr(mod, "main"):
        try:
            mod.main()
        except SystemExit:
            pass
        except Exception:
            # The script may fail for unrelated reasons (missing data files);
            # the parser is built before that and is all we need.
            pass

    ours = [p for p in seen
            if all(isinstance(getattr(a, "help", None), (str, type(None)))
                   for a in p._actions)]
    if not ours:
        # hydra builds its own parser with lazily-rendered help objects; the
        # hydra-driven scripts define no argparse help of their own.
        pytest.skip(f"{path.name} builds no plain ArgumentParser")
    for parser in ours:
        parser.format_help()      # raises on an unescaped %
