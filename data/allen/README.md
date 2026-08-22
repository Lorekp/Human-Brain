# Pinned Allen human cells

This directory contains the immutable Allen Cell Types inputs used by
`humanbrain.cells`. Each fitted model is bound to a human middle temporal gyrus
specimen, an exact downloaded configuration, a provenance response, and a
SHA-256 digest.

| Role | Specimen | Allen neuronal model | Model type |
| --- | --- | --- | --- |
| Fast language cell | `488392806` | `503612028` | 3 LIF + Afterspike Currents (LIF-ASC) |
| Recurrent-state cell | `508378649` | `523000114` | 5 LIF-R-ASC + Threshold Adaptation |

Both specimens are spiny human cortical cells from middle temporal gyrus. The
GLIF5 recurrent model adds fitted voltage reset, two after-spike currents, and
spike- and voltage-dependent threshold state that the GLIF3 model lacks.

## Files and checksums

| Model | Configuration | SHA-256 | Provenance |
| --- | --- | --- | --- |
| `503612028` | `503612028_neuron_config.json` | `57e20326d27146c3afbd412ca3ba0bcaee67a22f27729292b36666b9ff36c889` | `503612028_metadata.json` |
| `523000114` | `523000114_neuron_config.json` | `69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149` | `523000114_metadata.json` |

Original Allen endpoints:

- model `503612028`: [configuration](https://api.brain-map.org/api/v2/well_known_file_download/503679331), [model API](https://api.brain-map.org/neuronal_model/download/503612028);
- model `523000114`: [configuration](https://api.brain-map.org/api/v2/well_known_file_download/568808790), [model API](https://api.brain-map.org/neuronal_model/download/523000114).

## Runtime boundary

The source artifacts use SI units. `humanbrain.cells` explicitly converts them
to mV, ms, pA, pF, nS, and MΩ and ports the configured AllenSDK rules. Do not
edit a downloaded configuration in place: add a new pinned artifact, record its
provenance and digest, and update the protocol that admits it.

These are human-data-fitted point-cell prototypes. They are not
multicompartment reconstructions and do not satisfy the stricter Eyal
human-cell oracle described in the research plan. See the repository
[README](../../README.md) and [architecture](../../docs/architecture.md) for
the system-level claim boundary.
