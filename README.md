# PocketVec Human vs. *T. cruzi* Pocket Similarity

Structural bioinformatics pipeline that compares druggable binding pockets of human drug
targets with those of their *Trypanosoma cruzi* homologs using **PocketVec** binding-site
descriptors. Pocket pairs that look dissimilar point to binding sites where selective
anti-Chagas compounds might be found.

Human targets come from ChEMBL bioactivity data and ligand-bound PDB structures. *T. cruzi*
homologs come from AlphaFold DB models. Each pocket is encoded as a 128-dimensional PocketVec
fingerprint obtained by inverse docking with rDock, and human–parasite pocket pairs are compared
by cosine distance.

> **Showcase version.** This repository contains the pipeline code and a single example pair
> (human MAGL, PDB 3PE6, vs. its *T. cruzi* homolog, AlphaFold model AF-Q4CSM0-F1). Full
> screening results are intentionally not included.

---

## Pipeline overview

```
Target selection   ChEMBL 37 bioactivities ─► human single-protein targets ─► SIFTS PDB chains
                   ─► ligand-bound PDBs ─► ligand/pocket characterization ─► ranked targets

Homologs           human target ─► T. cruzi homolog (reciprocal BLAST best hit*) ─► AlphaFold DB model

PocketVec          structure retrieval ─► receptor preparation ─► pocket centroid
                   (crystallographic ligand, or fpocket + P2Rank) ─► rbcavity ─► rbdock × 128 molecules
                   ─► rank-based 128-d fingerprint

Comparison         human fp vs. T. cruzi fp ─► cosine distance ─► conserved / gray zone / selective
                   ─► AlphaFold pLDDT/PAE QC of the T. cruzi pocket
```
\* The reciprocal BLAST search itself is not scripted in this repository (see [TODO](#todo)).

| Stage | Approach | Scripts |
|---|---|---|
| **0. Target selection** | Extract ChEMBL 37 bioactivities, filter them, map human single-protein targets to PDB chains (SIFTS), characterize co-crystallized ligands and pockets, rank targets | `src/01–04`, `notebooks/chembl37_bioactivity_filtering.ipynb` |
| **1. Structure retrieval** | RCSB PDB for human structures; for *T. cruzi*, local AlphaFold models first, then the AlphaFold DB API, then direct file names v6→v1 | `src/05` → `utils/pipeline_steps.py` (`fetch_rcsb_structure`, `fetch_alphafold_structure`) |
| **2. Homolog identification** | Pre-BLAST exclusion of GPCR class A / nuclear-receptor candidates by Pfam and UniProt family; alternative UniProt-based lookup (OrthoDB → Pfam → UniRef50) | `utils/family_filter.py`, `utils/chembl_target_characterization.py` (`find_cruzi_homolog`) |
| **3. Pocket detection and descriptors** | Ligand-defined pocket when a crystallographic ligand passes the filters, otherwise fpocket + P2Rank rescoring; volume, druggability score and residue composition | `utils/pipeline_steps.py`, `utils/ligand_characterization.py` |
| **4. PocketVec fingerprints** | rbcavity around the pocket centroid, rigid-ligand rbdock of 128 lead-like molecules, best score per molecule converted to ranks | `utils/pipeline_steps.py` (`run_pocketvec`), `utils/pocketvec_utils.py` |
| **5. Pocket comparison** | Cosine distance between paired fingerprints, three-band classification and plots; pocket-level pLDDT/PAE QC of the AlphaFold models | `src/05` (`compare_pair`), `src/06`, `src/07` |
| **Case study** | MAGL: catalytic-triad-anchored superposition (cealign), pocket subregions and oxyanion hole, substrate tunnel | `src/case_studies/magl/`, `notebooks/magl_superposition_interactive.ipynb` |

Stages 1 and 3–5 run per target inside one orchestrator (`05_run_pocketvec_batch.py`) for both
sides of each pair. A failed target is logged and skipped, finished fingerprints are not
recomputed, and two pairs that share a structure reuse the same result.

---

## Methods and design decisions

All values below are taken from the code; paths and tool locations are set in
[`config/config.yaml`](config/config.yaml).

| Setting | Value | Notes |
|---|---|---|
| Valid crystallographic ligand | HETATM, not an amino acid or common modification, n<sub>C</sub> > 6, solvent accessibility < 0.4, buriedness > 15 | Solvent accessibility = SASA in complex / SASA isolated (Shrake–Rupley). Buriedness = protein heavy atoms within 4.5 Å of the ligand. Waters, ions and crystallization additives are excluded. |
| Pocket centroid (human) | Best-scoring valid ligand | If the same ligand is present in several listed chains, the centroid is computed from all copies. |
| Pocket centroid (no valid ligand, all AlphaFold models) | fpocket pocket with the highest druggability score | P2Rank rescoring is computed alongside; `criterion="prank"` is available as an alternative. |
| Residue composition | Hydrophobic / polar / charged / aromatic counts and fractions within 5 Å of the ligand | Computed for ligand-defined pockets in `ligand_characterization.py`. |
| Receptor preparation | First model → selected chains → ligands and waters removed → hydrogens removed and alternate locations resolved (biobb `check_structure`) → MOL2 | Uses MOE if configured, otherwise Open Babel. All ligands are removed, so docking runs against the empty cavity. |
| Cavity definition | rDock reference-ligand site mapper, radius 12 Å around the centroid | `pocketvec.radius` in the config. |
| Docking | rbdock, rigid ligands (`DIHEDRAL_MODE FIXED`), 25 runs per molecule, seed 42 | `pocketvec.nruns` in the config. |
| Ligand library | 128 lead-like molecules (`TOP_128_rDock_LLM.sdf`) | Same library and molecule order as PocketVec. |
| Fingerprint | Best `SCORE.INTER` per molecule → rank (`method="max"`); positive scores pushed to the lowest ranks | 128 dimensions, one per library molecule. |
| Pair similarity | Cosine distance (`scipy.spatial.distance.cosine`) | Classification: < 0.17 conserved (off-target risk), 0.17–0.20 gray zone, ≥ 0.20 selective (CLI-adjustable). |
| AlphaFold pocket QC | Pocket pLDDT < 70, global − pocket pLDDT > 15, or intra-pocket PAE > 5 Å | OK / REVISAR / DESCARTAR, using the fpocket residues of the selected pocket. |
| PDB selection (target screening) | X-ray or EM, resolution ≤ 3.0 Å, up to 3 PDB entries per target | Targets are ranked by a composite of druggability, ligand–pocket overlap, centroid distance and volume (`04_rank_targets.py`). |

More detail in [`docs/methodology.md`](docs/methodology.md).

---

## External tools

| Tool | Version used | Role |
|---|---|---|
| [fpocket](https://github.com/Discngine/fpocket) | 4.2.3 | Pocket detection and druggability score |
| [P2Rank](https://github.com/rdk/p2rank) | 2.5.1 (Java ≥ 17) | Pocket rescoring |
| [rDock](https://github.com/CBDD/rDock) | 25.07-alpha (precompiled build) | Cavity definition (`rbcavity`) and docking (`rbdock`); `environment.yml` installs the bioconda build, which is a different version |
| [Open Babel](https://github.com/openbabel/openbabel) | 3.1.1 | Format conversion, centroid files, reading docking poses |
| biobb_structure_checking | — | `check_structure` for hydrogen and alternate-location handling |
| BLAST+ | not recorded | Reciprocal best-hit search (run outside this code base, see TODO) |
| PyMOL (open source) | — | MAGL case study only |
| MOE | optional | Receptor preparation; the pipeline falls back to Open Babel without it |

---

## Installation

```bash
conda env create -f environment.yml
conda activate pocketvec-tcruzi
```

Then point `config/config.yaml` to the external resources:

- **P2Rank:** download the 2.5.1 release from GitHub and set `tools.prank` to its `prank`
  launcher.
- **rDock:** set `tools.rdock_root` to a directory that contains `bin/`, `lib/` and `dock.prm`
  (rDock's standard protocol file, `data/scripts/dock.prm` in the rDock distribution).
- **PocketVec ligand library:** this repository does not include it. Get
  `TOP_128_rDock_LLM.sdf` and its molecule-order file `order/TOP_128_rDock_LLM.pkl` from the
  original PocketVec distribution and place them under `data/libs/`.
- **Target selection only:** you also need the ChEMBL 37 SQLite dump (`chembl.db`) and the
  wwPDB/SIFTS bulk files (`pdb_chain_uniprot.csv.gz`, `entries.idx`).

Tools given as bare names (`fpocket`, `obabel`, `check_structure`) are looked up in `PATH`.
Each one can also be overridden with an environment variable (`POCKETVEC_FPOCKET`,
`POCKETVEC_PRANK`, `POCKETVEC_OBABEL`, `POCKETVEC_CHECK_STRUCTURE`, `POCKETVEC_RADIUS`,
`POCKETVEC_NRUNS`).

---

## Usage

**Example pair:** recomputes the human–*T. cruzi* cosine distance from the bundled
fingerprints, then runs the analysis and the pLDDT QC. No docking and no network access:

```bash
scripts/run_pipeline.sh --example        # outputs in results/example/
```

**Full pipeline:**

```bash
scripts/run_pipeline.sh --selection      # 01–04: ChEMBL → ranked human targets
# build data/targets.csv (one row per human–T. cruzi pair, see below)
scripts/run_pipeline.sh --pocketvec      # 05–07: PocketVec batch, analysis, QC
```

**Single stages:**

```bash
python src/05_run_pocketvec_batch.py --target-ids 1,3    # or --target-names MAGL
python src/06_analyze_selectivity.py --threshold 0.17 --gray-zone 0.20
python src/07_qc_plddt_pocket.py --no-download
python src/case_studies/magl/superpose_magl.py           # requires prepared MAGL structures
```

To run on another dataset, merge a second YAML over the defaults with
`POCKETVEC_CONFIG=path/to/override.yaml` (see `config/example.yaml`).

The batch reads one row per pair from `paths.targets_csv`. The columns it uses are
`target_id`, `target_name`, `human_pdb_code`, `human_chain`, `ligand_chains`,
`ligands_to_remove`, `human_uniprot`, `cruzi_uniprot`, `cruzi_af_status` (only `found` rows
are docked), `cruzi_af_code` and `cruzi_name`. See
[`data/example/targets_example.csv`](data/example/targets_example.csv).

---

## TODO

These parts of the pipeline exist in the project but are not implemented as code in this
repository:

- **Reciprocal BLAST best hit (Stage 2).** The forward search (human → *T. cruzi* proteome),
  the reverse search and the best-hit reciprocity check were run outside this code base. Add a
  script with the `blastp` / `makeblastdb` commands, thresholds and proteome versions.
- **Pfam domain-architecture validation of homolog pairs.** `family_filter.py` only excludes
  GPCR class A and nuclear-receptor candidates before BLAST. A pair-level comparison of domain
  architectures is not coded.
- **Building `targets.csv`.** Merging the ranked targets (`04_rank_targets.py`) with the
  homolog search into the per-pair input table was done by hand. Likewise, the summary table
  used by the QC script (`paths.qc_targets_csv`) has no generating script.
- **Residue composition of *T. cruzi* pockets.** The hydrophobic/polar/charged/aromatic
  composition is computed only for ligand-defined (human) pockets.
- **Source URL of the PocketVec ligand library** (`TOP_128_rDock_LLM.sdf` and its order file).

---

## Repository structure

```
├── README.md
├── environment.yml
├── config/
│   ├── config.yaml          paths, external tools, PocketVec parameters
│   └── example.yaml         overrides for the bundled example pair
├── src/
│   ├── 01_extract_chembl.py            ChEMBL 37 bioactivity export (SQLite)
│   ├── 02_map_targets_to_pdb.py        targets → UniProt → PDB chains (SIFTS, entries.idx)
│   ├── 03_characterize_pdb_ligands.py  ligand/pocket characterization of candidate PDBs
│   ├── 04_rank_targets.py              target-level aggregation and ranking
│   ├── 05_run_pocketvec_batch.py       structures, pockets, PocketVec fingerprints, cosine distance
│   ├── 06_analyze_selectivity.py       selectivity classification and plots
│   ├── 07_qc_plddt_pocket.py           pocket-level pLDDT/PAE QC of AlphaFold models
│   ├── utils/
│   │   ├── config.py                   loads config.yaml, per-protein output paths
│   │   ├── pipeline_steps.py           per-protein preparation, pockets, PocketVec
│   │   ├── pocketvec_utils.py          rDock / fpocket / P2Rank helpers (adapted from PocketVec)
│   │   ├── ligand_characterization.py  ligand filters, pocket descriptors, residue composition
│   │   ├── chembl_target_characterization.py   ChEMBL/UniProt/PDBe/AlphaFold DB lookups
│   │   ├── screening_classify.py       docking-suitability tiers for target screening
│   │   └── family_filter.py            Pfam/UniProt family pre-filter for homolog search
│   └── case_studies/magl/              MAGL superposition, subregions, tunnel, renders
├── scripts/run_pipeline.sh
├── notebooks/
│   ├── chembl37_bioactivity_filtering.ipynb
│   └── magl_superposition_interactive.ipynb
├── data/example/            MAGL pair: input structures, pocket files, fingerprints
└── docs/methodology.md
```

The pipeline writes its outputs (downloaded structures, prepared receptors, fpocket/P2Rank
output, docking results, logs and plots) to `results/`, which is excluded from version
control. The same applies to the ChEMBL dump, bulk PDB files, the ligand library and
external tools.

---

## Acknowledgements

The PocketVec descriptor, its 128-molecule lead-like library and the helper routines adapted in
`src/utils/pocketvec_utils.py` come from the original PocketVec work by Comajuncosa-Creus *et al.*

---

## Author

**Luz Sommariva** — Biotechnology · Bioinformatics · Computational Chemistry
