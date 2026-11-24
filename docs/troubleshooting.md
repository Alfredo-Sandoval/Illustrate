# Troubleshooting

## Installation

- **`No matching distribution found for illustrate-py`.** Use the [Git install command](getting-started.md#install-on-macos) until the package is published on PyPI.
- **`illustrate-py: command not found` or `No module named 'PySide6'`.** Run the command in the Python environment where you installed Illustrate. The desktop app needs the `gui` extra. From a checkout, `make start` sets up and launches the managed app.
- **`Could not download micromamba`.** The source installer needs access to `micro.mamba.pm` when it cannot find Mamba or Micromamba. Install one of them, or use the pip installation in an existing Python environment.

## Structures and rendering

- **`no such file, and not a four-character PDB ID`.** Check the local path or the four-character ID. Fetch requires a structure available in PDB format.
- **`Scale is too high for these atom radii`.** Lower `--scale` or **Scale**, or reduce the largest radius in the rule list. Even a rule that matches no atoms counts toward the limit.
- **Missing atoms.** Check the [selection rules](guide/selection-rules.md) for an earlier match with zero radius or a missing rule for the atom’s record type. The reader uses the PDB’s biomolecule 1 records when present. It stores at most 500 BIOMT transforms and rejects excess transforms in strict mode.
- **Malformed input.** Pass `--strict` in the CLI to report malformed command cards and PDB fields the parser would otherwise tolerate. In the PDB reader, unreadable coordinate fields are checked for atoms retained by the rules.
- **`Matches` shows `-`.** No match counts are available for the current rules. Load a structure and click **Render** to classify it.
- **Settings refer to a missing file.** Supply the PDB path as a separate CLI argument to override the one in the JSON. In the GUI, [save settings](guide/gui.md#save-and-export) after opening the intended structure.
