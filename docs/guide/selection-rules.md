# Selection Rules

Selection rules determine atom color and radius assignment. Rules are evaluated top-to-bottom, and the first matching rule is used.

## Rule fields

| Field | Type | Meaning |
|------|------|---------|
| `record_name` | `str` | Compared against PDB columns 1-6 (`ATOM  ` / `HETATM`) |
| `descriptor` | `str` | 10-character pattern compared against PDB columns 13-22 |
| `res_low` | `int` | Minimum residue number (inclusive) |
| `res_high` | `int` | Maximum residue number (inclusive) |
| `color` | `tuple[float, float, float]` | RGB in `[0.0, 1.0]` |
| `radius` | `float` | Radius in angstroms (`0.0` hides matched atoms) |

## Descriptor matching

Descriptor matching is literal per character over a 10-character window.

- Source window: PDB columns 13-22.
- Wildcard: `-` matches any character.
- Short descriptors are right-padded with `-` by the renderer before matching.

Why edit descriptors:

- Target specific chains (for example `---------A` for chain A).
- Target specific residues or ligands by name pattern (for example `-----HOH--` for water).
- Isolate atom-name patterns (for example carbon-only or sulfur-only rules) before broader catch-all rules.

Examples:

```python
"----------"   # wildcard catch-all
"-----HOH--"   # HOH residues (used in default rules)
"---------A"   # chain A (column 22)
"FE---HEM--"   # FE in HEM (used in default rules)
```

## Ordering matters

If a broad rule appears early, specific rules below it will never fire.

Typical safe ordering:

1. Exclusions (`radius=0.0`) such as water/hydrogens.
2. Specific residue/atom rules.
3. Chain-level rules.
4. Final catch-all (`----------`).

## Minimal examples

### Hide water

```python
SelectionRule("HETATM", "-----HOH--", 0, 9999, (0.5, 0.5, 0.5), 0.0)
```

### Two-chain coloring + catch-all

```python
SelectionRule("ATOM  ", "---------A", 0, 9999, (1.0, 0.5, 0.5), 1.5)
SelectionRule("ATOM  ", "---------B", 0, 9999, (0.5, 0.7, 1.0), 1.5)
SelectionRule("ATOM  ", "----------", 0, 9999, (0.9, 0.8, 0.7), 1.5)
```

### Residue-range highlight

```python
SelectionRule("ATOM  ", "----------", 50, 100, (1.0, 0.9, 0.0), 1.5)
SelectionRule("ATOM  ", "----------", 0, 9999, (0.7, 0.7, 0.7), 1.5)
```

## GUI behavior

In the desktop GUI, descriptor is editable directly in the `Descriptor` column.

The **Matches** column updates after rendering and reports per-rule atom counts for the currently loaded structure.
