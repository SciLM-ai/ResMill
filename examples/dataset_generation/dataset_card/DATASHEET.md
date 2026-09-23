# Datasheet for *SiliciclasticReservoirs*

Following the Datasheet-for-Datasets template (Gebru et al., 2018, [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)).

---

## 1. Motivation

**For what purpose was the dataset created?**
To enable training of conditional generative models (flow matching, diffusion, autoregressive, etc.) of subsurface 3D siliciclastic reservoir geology under interpretable physical conditioning. Existing public 3D-reservoir datasets are either tiny (hundreds of cubes) or proprietary; this dataset addresses that gap with a million synthetic samples spanning eight architectural classes drawn from the standard fluvial-reservoir taxonomy (Pyrcz & Deutsch, 2002).

Specifically, the dataset is designed to support:
- Conditional generative modeling of facies, porosity, and permeability conditioned on net-to-gross, sinuosity, channel geometry, etc.
- Text-to-3D research using natural-language captions
- Architectural classification across the 8 reservoir families
- Data augmentation for sparse real-reservoir datasets via transfer learning

**Who created the dataset and on behalf of which entity?**
Created by [Anonymized for double-blind review] as part of research on AI-driven subsurface modeling.

**Who funded the creation of the dataset?**
[Anonymized for double-blind review]

---

## 2. Composition

**What do the instances represent?**
Each instance is one synthetic 3D reservoir geology cube of shape `(64, 64, 32)` voxels, cut at a random position out of a simulated `(128, 128, 64)` volume, representing a small subsurface volume (6.4 km × 6.4 km × 32 m for lobes; 640 m × 640 m × 32 m for channels and delta). Each cube carries:
- `facies` (binary 0/1, sand vs. mud)
- `facies_alluvsim` (6-class architectural detail)
- `poro` (continuous porosity per voxel)
- `perm` (continuous permeability per voxel, in mD)

**How many instances are there in total?**
**1,000,000 instances**, broken down by architectural family:

| `layer_type` | count |
|---|---|
| `lobe` | 200,000 |
| `channel:PV_SHOESTRING` | 100,000 |
| `channel:CB_LABYRINTH` | 100,000 |
| `channel:CB_JIGSAW` | 150,000 |
| `channel:SH_DISTAL` | 100,000 |
| `channel:SH_PROXIMAL` | 100,000 |
| `channel:MEANDER_OXBOW` | 100,000 |
| `delta` | 150,000 |

**Does the dataset contain all possible instances or is it a sample?**
The dataset is a Sobol quasi-random sample from a parameterized geological-rule space. The full parameter space is continuous and high-dimensional; this dataset is a **statistically representative quasi-random subset** sized for ML training. Sobol coverage is uniform within each layer-type's parameter ranges (see `params.parquet` for full ranges).

**What data does each instance consist of?**
- **Voxel data** (4 arrays of shape `(64, 64, 32)`):
  - `facies` (`int8`, binary 0/1)
  - `facies_alluvsim` (`int8`, codes -1 to 4)
  - `poro` (`float16`, [0, 0.5])
  - `perm` (`float16`, [0, 60000] mD)
- **Slim parquet row** (~12 columns): training-time conditioning parameters
- **Full parquet row** (34 to 108 columns): all physics parameters plus the window origin inside the simulated volume, for reproducibility
- **Caption**: human-readable English description (~30 words)

**Is there a label or target associated with each instance?**
Each instance is self-labeled by its layer_type (categorical) and slim-parquet conditioning (continuous). For typical generative-modeling use, there is no separate "target": the model learns to generate cubes conditioned on parameters.

**Is any information missing from individual instances?**
Family-specific parameters (e.g., `asp` is lobe-only; `trunk_length_fraction` is delta-only) are NULL in non-applicable rows. This is by design — different architectures have different physics knobs.

**Are relationships between individual instances made explicit?**
No. Instances are statistically independent draws within their layer-type. Rows with the same Sobol seed across different layer-types are NOT meaningfully related.

**Are there recommended data splits?**
Yes — `splits/{train,validation,test}.parquet` provide a deterministic 90 / 5 / 5 split stratified by `layer_type`, generated with seed 42. Reviewers and benchmarks should use these splits.

**Are there any errors, sources of noise, or redundancies in the dataset?**
- The simulation engine has known approximations (see *Geological caveats* in README): fluvial channels split but do not re-merge; per-cell properties follow analytic ramps with a correlated texture rather than fully heterogeneous Gaussian fields; FACIES_PROPS values are textbook midpoints rather than from any specific reservoir analog.
- Float16 storage of `poro` / `perm` introduces ~0.05% quantization error vs. the engine's float32 internal representation.
- No duplicates within or across splits — Sobol guarantees no exact repeats; very-near-neighbor draws are possible but rare at 1M scale in a ~10-dim space.

**Is the dataset self-contained, or does it rely on external resources?**
**Fully self-contained.** No external links, no external assets. The engine code is [ResMill](https://anonymous.4open.science/r/ResMill-7377) is independent open-source software; users do not need to run it to use this dataset.

**Does the dataset contain data that might be considered confidential?**
No. All data is synthetic.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
No. Synthetic geological data.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**
Each instance was generated by deterministic execution of the custom simulation engine, [ResMill](https://anonymous.4open.science/r/ResMill-7377), a Python + Numba implementation of rule-based sedimentological simulation. The engine ports the streamline-based fluvial architecture from Alluvsim (Pyrcz, 2003; Pyrcz & Deutsch, 2002) into a Python-native framework with extensions for delta architectures and per-cell porosity/permeability fields.

**What mechanisms or procedures were used to collect the data?**
Sobol quasi-random sampling of the parameter space (`scipy.stats.qmc.Sobol`) generated 1M parameter sets; each set was passed to the engine, which produced one `(128, 128, 64)` volume, and one `(64, 64, 32)` window was cut out of each volume at a seeded random position. Generation ran as eight Slurm jobs (one per architecture, 96 to 432 ranks each, one rank per core) on TACC Vista's Grace CPU nodes, 33 node-hours in total; each rank processed a deterministic stripe of the shuffled job list.

**If the dataset is a sample, what was the sampling strategy?**
- Per layer-type Sobol sampling (`scrambled=True`) with deterministic seeds (master_seed=42; per-layer section_seed = (43 × 10007) + lt_id)
- Stratified by layer-type via the per-architecture `count` parameter in each config
- Some parameters use coupled / shared / log-uniform sampling distributions documented in the engine config files

**Who was involved in the data collection process?**
[Anonymized for double-blind review].

**Over what time-frame was the data collected?**
Generated in September 2026 in a single four-hour compute window; windows, combined shards and splits were produced the same day.

**Were any ethical review processes conducted?**
Not applicable — no human subjects, no real-world data, no proprietary information.

**Did you collect the data from individuals, or via a third party?**
No. Synthetic data generated by deterministic simulation.

**Were the individuals notified of data collection? Did they consent?**
N/A.

**Has an analysis of the potential impact of the dataset and its use been conducted?**
Limited. Geological synthetic data has low ethical risk profile: no demographic skew, no privacy implications, no surveillance enablement. The primary "use" risk is researchers training models on synthetic data and over-claiming generalization to real reservoirs without separate validation. The README and this datasheet flag this explicitly.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes:

- **Windowing**: Each cube is a `(64, 64, 32)` window of the engine's `(128, 128, 64)` volume at a seeded random origin (x and y anywhere, z between 1 and 31 so the window never contains the simulated floor or roof), redrawn while the window holds no sand. The origin and the source volume are recorded in `crop_x0, crop_y0, crop_z0, crop_seed, source_shard, source_row`.
- **Quantization**: `poro` and `perm` are clipped to `[0, 0.5]` and `[0, 60000]` then cast to `float16` for storage compactness. `facies` and `facies_alluvsim` are stored as `int8`.
- **Realized statistics**: `ntg`, `poro_ave`, `perm_ave` and the caption are recomputed on the stored window, so they reflect what is actually in the saved cube (not the requested target).
- **Captions**: Generated by a deterministic template at write-time from the realized parameters.
- **Failure handling**: Any sample whose engine call raised an exception was logged to `failures_r*.jsonl` and excluded; all 1M samples in this release succeeded with zero exceptions.

**Was the "raw" data saved?**
Yes — the full physics parameters are preserved in `params.parquet` per shard, the (seed, params) pair regenerates the `(128, 128, 64)` volume bit-for-bit with the public engine code, and the `crop_*` columns locate the cube inside it. The simulated volumes themselves (5.8 TB) are kept by the authors and are not part of the release.

**Is the software used to preprocess/clean/label the data available?**
Yes — the entire pipeline (engine, sampler, dataset writer, splitter) is in the open-source repository, [ResMill](https://anonymous.4open.science/r/ResMill-7377).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Dataset has been used to train a flow matching based foundation model, [ResFlow](https://anonymous.4open.science/r/ResFlow-5654), for generation of all types of siliciclastic reservoirs.

**Is there a repository that links to any or all papers or systems that use the dataset?**
Will be maintained at the HuggingFace dataset page once papers using it are published.

**What (other) tasks could the dataset be used for?**
- Conditional generative modeling (flow matching, diffusion, GAN) of 3D facies + petrophysical properties
- Architectural classification (8-class reservoir-architecture identification from voxel cubes)
- Self-supervised representation learning on 3D voxel data
- Inverse problem benchmarks: given partial observations (well logs, seismic-resolution downsampling), reconstruct the full cube
- Transfer learning to small real-reservoir datasets (synthetic pretraining + real-data fine-tuning)
- Pretraining data for foundation models in subsurface science

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**

Yes — important caveats:
- **Fluvial channels split but do not re-merge**; only the delta's distributary branches may rejoin. Real braided rivers have channels that split AND re-merge. For pure facies generative modeling this is invisible; for downstream flow-simulation evaluation it matters.
- **Per-cell `poro` / `perm`** follow a Walker-1992 upward-fining ramp per channel event, per-event Kozeny-Carman-coupled draws, a per-realization multiplier and a correlated Gaussian texture inside sand bodies; they are not conditioned to any real reservoir.
- **Cubes are windows**: sand bodies are cut by the cube faces the way a modelled interval cuts geology; a body truncated at a face continues in the simulated volume.
- **FACIES_PROPS base values are textbook midpoints** for fluvial reservoirs (Pyrcz & Deutsch 2002, Allen 1965); not specific to any real reservoir analog. Per-realization multiplicative scaling spans plausible ranges but the specific values shouldn't be interpreted as quantitative predictions for a particular basin.

**Are there tasks for which the dataset should not be used?**
- **Quantitative forecasts of specific real reservoirs.** Synthetic-only training without real-data fine-tuning is unlikely to produce calibrated predictions.
- **Decisions about real-world drilling, hydraulic-fracturing, or carbon-storage site selection** without independent geological validation.
- **As a stand-in for paleogeographic field work.** This dataset is designed to span an architectural taxonomy; it does not capture the regional / climatic / tectonic context that drives real-world geology.

---

## 6. Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which it was created?**
Yes — released publicly under CC-BY-4.0 on HuggingFace.

**How will the dataset be distributed?**
HuggingFace Datasets (anonymized; URL to be disclosed at camera-ready). Total size about 733 GB across 8 layer-type subdirectories.

**When will the dataset be distributed?**
September 2026.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). Attribution required; commercial use permitted; modifications permitted.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
No.

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**
No.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
[Anonymized for double-blind review], with hosting on HuggingFace.

**How can the owner/curator/manager of the dataset be contacted?**
Through the HuggingFace dataset issue tracker, or through the GitHub repository (anonymized; URL to be disclosed at camera-ready).

**Is there an erratum?**
None. Errata will be posted on the HuggingFace dataset page; earlier revisions of the repository remain accessible through its revision history.

**Will the dataset be updated?**
Possibly — additional architectures, larger samples, alternative grid resolutions. Updates will be posted as new HuggingFace dataset versions; the current 1M-sample release will remain accessible by version pinning.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**
N/A — no human data.

**Will older versions of the dataset continue to be supported/hosted/maintained?**
Yes — HuggingFace's revision system preserves old versions.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes — the engine code is open-source [ResMill](https://anonymous.4open.science/r/ResMill-7377). Anyone can:
- Generate additional samples with their own configs
- Modify the engine for new architectures (e.g., aeolian, carbonate)
- Submit pull requests to upstream new architectures
- Publish derivative datasets under their own HuggingFace namespaces

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2018). *Datasheets for Datasets*. arXiv:1803.09010.
- Pyrcz, M. J. (2003). *Stochastic Surface-based Modeling of Turbidite Lobes*. PhD dissertation, University of Alberta.
- Pyrcz, M. J., & Deutsch, C. V. (2002). *User Guide to the Alluvsim Program*. Centre for Computational Geostatistics.
- Walker, R. G. (1992). *Facies Models: Response to Sea Level Change*. Geological Association of Canada.
- Allen, J. R. L. (1965). *A review of the origin and characteristics of recent alluvial sediments*. Sedimentology.
