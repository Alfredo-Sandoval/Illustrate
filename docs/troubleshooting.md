# Troubleshooting

## Installation

**`Could not download micromamba`**

Your network may be blocking `https://micro.mamba.pm/`.

Try one of these:

- Re-run behind an unrestricted network/VPN.
- Install micromamba manually, then retry:

```bash
make install-desktop
```

**`Missing dependency: curl`** or **`Missing dependency: tar`**

The automatic bootstrap requires `curl` and `tar`. Install those tools from your OS package manager, then run:

```bash
make install-desktop
```

**`ModuleNotFoundError: No module named 'illustrate'`** (or `PySide6`)

Repair the managed desktop runtime:

```bash
make install-desktop
```

Then launch:

```bash
make start
```

**`illustrate-py: command not found`**

If you are using the managed environment path, reinstall runtime first:

```bash
make install-desktop
```

If you are using a custom Python environment instead:

```bash
pip install -e .
```

You can also invoke the module directly if your environment is configured:

```bash
python -m illustrate.cli
```

---

## CLI

**`no command stream on stdin`**

The CLI reads from stdin. Use redirect or a pipe:

```bash
illustrate-py < file.inp
```

**`parse error: ...`**

The command stream is invalid. For stricter validation:

```bash
illustrate-py --strict-input < file.inp
```

**`PDB read/classification error: ...`**

The referenced PDB path is wrong for the current working directory, or the PDB content is invalid.

**`render error: atoms radius * scale > 100`**

Reduce atom radii and/or transform scale so `radius * scale <= 100`.

---

## GUI

**Selection Rules `Matches` column shows `-`**

Match counts are computed after rendering. Load a structure and run **Render**.

**`Fetch failed: ...`**

The entered PDB ID failed validation or download. Use a 4-character alphanumeric ID and retry.

---

## Web API

**`400 Unknown pdb_id: ...` from `/api/render`**

`/api/render` requires a valid upload token from `/api/upload-pdb` or `/api/fetch-pdb`.

**`400 Unsupported output format: ...` from `/api/render`**

Allowed output formats are PNG and PPM aliases:

- `png`
- `image/png`
- `ppm`
- `pnm`
- `image/x-portable-pixmap`

**`422 invalid render parameters: ...` from `/api/render`**

Payload shape is valid JSON, but one or more render fields are invalid (for example bad `autocenter` or malformed transform values).
