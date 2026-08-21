# Pinned Allen human cells

The fast language cell is fitted to Allen Cell Types specimen `488392806`, a
spiny human layer-2 neuron from middle temporal gyrus, and neuronal model
`503612028`, “3 LIF + Afterspike Currents (LIF-ASC).”

- Configuration: `503612028_neuron_config.json`
- Configuration SHA-256:
  `57e20326d27146c3afbd412ca3ba0bcaee67a22f27729292b36666b9ff36c889`
- Provenance response: `503612028_metadata.json`
- Configuration URL:
  `https://api.brain-map.org/api/v2/well_known_file_download/503679331`
- Model API: `https://api.brain-map.org/neuronal_model/download/503612028`

The source artifact is in SI units. `humanbrain.cells` performs an explicit
conversion to mV, ms, pA, pF, nS, and MΩ and ports the configured AllenSDK
rules.

The recurrent-state cell is fitted to a second spiny human middle-temporal-
gyrus specimen, `508378649`, using model `523000114`, “5 LIF-R-ASC +
Threshold Adaptation.” It supplies the fitted voltage reset, two after-spike
currents, and spike- and voltage-dependent threshold state that GLIF3 lacks.

- Configuration: `523000114_neuron_config.json`
- Configuration SHA-256:
  `69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149`
- Provenance manifest: `523000114_metadata.json`
- Configuration URL:
  `https://api.brain-map.org/api/v2/well_known_file_download/568808790`

Both are human-data-fitted point-cell scale prototypes. They are not the
strict multicompartment Eyal human-cell oracle described in the research plan.
