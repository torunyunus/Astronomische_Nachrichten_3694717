# Data dictionary

The combined randomized SDSS DR17 analysis table is expected to contain the following source-level fields.

| Field | Description |
|---|---|
| `objid` | SDSS photometric object identifier |
| `ra`, `dec` | Equatorial coordinates in degrees |
| `u`, `g`, `r`, `i`, `z` | SDSS five-band photometric magnitudes |
| `Err_u`, `Err_g`, `Err_r`, `Err_i`, `Err_z` | Photometric magnitude uncertainties |
| `specobjid` | SDSS spectroscopic object identifier |
| `class` | Spectroscopic class label: GALAXY, STAR, or QSO |
| `redshift` | Spectroscopic redshift |
| `zErr` | Spectroscopic redshift uncertainty |
| `plate`, `mjd`, `fiberID` | Spectroscopic observation identifiers |

## Derived colour features

Seven colour indices are constructed directly from the five magnitudes:

- `u-g = u - g`
- `g-r = g - r`
- `r-i = r - i`
- `i-z = i - z`
- `u-r = u - r`
- `g-i = g - i`
- `r-z = r - z`

The colour features are therefore deterministic transformations of the magnitude features. Grouped permutation analysis is used when interpreting magnitude and colour blocks jointly.

## Class encoding

The analysis scripts map the spectroscopic classes as follows:

- GALAXY → 0
- STAR → 1
- QSO → 2

## Photometric uncertainty

`Err_r` is used for the main photometric-uncertainty stratification. Approximate r-band signal-to-noise is calculated as:

`S/N ≈ 1.0857 / Err_r`

Photometric uncertainty columns are used for sensitivity analysis and are not automatically added to the original F1–F7 classifier input configurations.
