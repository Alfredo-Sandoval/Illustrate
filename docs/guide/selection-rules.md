# Selection rules

The first rule that matches a PDB atom assigns its RGB color and sphere radius. A zero radius hides the atom. Atoms without a match are left out.

| Field | Meaning |
| --- | --- |
| `record_name` | PDB columns 1–6, such as `ATOM` or `HETATM`. Short names are space-padded. |
| `descriptor` | Pattern for PDB columns 13–22. Each `-` matches one character. Short patterns are padded on the right with `-`. Spaces match literally. |
| `res_low`, `res_high` | Inclusive residue number range. |
| `color` | RGB components from 0 to 1. |
| `radius` | Sphere radius in ångströms. |

Put exclusions and specific rules before broader rules for the same record type. The following rules hide water, color chain A pink, and give other standard atoms a neutral color.

```python
from illustrate import SelectionRule

rules = [
    SelectionRule("HETATM", "-----HOH--", 0, 9999, (0, 0, 0), 0),
    SelectionRule("ATOM", "---------A", 0, 9999, (1.0, 0.5, 0.5), 1.5),
    SelectionRule("ATOM", "----------", 0, 9999, (0.7, 0.7, 0.7), 1.5),
]
```

The last rule matches only `ATOM` records in the given residue range. Add a separate `HETATM` rule if you also want to show ligands.

The desktop rule table exposes **Atom** (columns 13–16), **Residue** (18–20), and **Chain** (22) as separate descriptor fields. **Any** fills a field with wildcards. **Matches** counts the atoms retained by classification, before applying assembly copies or rendering.
