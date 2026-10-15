# Methodology

This document describes what each stage of the pipeline does, using the parameters set in the
code and in `config/config.yaml`. It reports no results.

---

## 0. Target selection (ChEMBL → PDB)

**Bioactivity extraction (`01_extract_chembl.py`).** Bioactivities are exported from a local
ChEMBL 37 SQLite dump with a single SQL query, in chunks.

**Bioactivity curation (`notebooks/chembl37_bioactivity_filtering.ipynb`).** Assays reported in
`%` are discarded. Only IC50, EC50, AC50, Ki, Kd and Potency are kept, and values are
standardized to nM. Duplicate compound–target measurements are collapsed to their median.
Activities are then binarized: active if the relation is `=` or `<` and the value is ≤ 10 µM;
inactive if the relation is `=` or `>` and the value is ≥ 20 µM; anything else is
indeterminate. When duplicates disagree, Ki/Kd measurements are preferred.

**Target → PDB mapping (`02_map_targets_to_pdb.py`).** Human single-protein targets are mapped
to UniProt accessions through the local ChEMBL tables, and then to PDB chains through the
SIFTS bulk file (`pdb_chain_uniprot.csv.gz`). Structures are kept if they are X-ray or
electron microscopy with resolution ≤ 3.0 Å (from `entries.idx`), and at most 3 PDB entries
per target are retained, ranked by resolution.

**Ligand and pocket characterization (`03_characterize_pdb_ligands.py`).** Each candidate PDB
entry is downloaded and passed to `characterize_holo_pdb_ligand_pockets()` (see Stage 3). Each
ligand–pocket pair is then placed in one of four docking-suitability tiers
(`utils/screening_classify.py`):

| Tier | fpocket druggability | Distance pocket–ligand centroid | Residue Jaccard (5 Å) |
|---|---|---|---|
| Favorable | ≥ 0.5 | ≤ 6 Å | ≥ 0.3 |
| Difficult | < 0.3 | > 10 Å | < 0.15 (any one of the three) |
| Intermediate | otherwise | | |
| Discard | ligand fails the validity filters | | |

**Target ranking (`04_rank_targets.py`).** For each target, the best ligand–pocket pair is kept:
best tier first, then highest composite score.

```
composite = 0.40·druggability + 0.25·jaccard_5Å + 0.20·clip(1 − d/10 Å, 0, 1) + 0.15·norm_volume
```

`norm_volume` equals 1 for pocket volumes between 300 and 5000 Å³ and decreases linearly
outside that range.

---

## 1. Structure retrieval

- **Human:** the PDB file is downloaded from RCSB, and only the first model and the selected
  chain(s) are kept.
- ***T. cruzi*:** a local model in `paths.local_alphafold_dirs` is used first (for predictions
  that are not in AlphaFold DB). Otherwise, the current model URL is requested from the
  AlphaFold DB API. If that fails, file names `AF-<UniProt>-F1-model_v6…v1.pdb` are tried
  directly.

## 2. Homolog identification

- **Family pre-filter (`utils/family_filter.py`).** Before BLAST, candidates are excluded if
  UniProt annotates them as GPCR class A (Pfam PF00001, or the "G protein-coupled receptor 1
  family" similarity comment) or as nuclear hormone receptors (Pfam PF00104 / PF00105, or the
  "nuclear hormone receptor family" comment).
- **UniProt-based lookup (`find_cruzi_homolog` in `utils/chembl_target_characterization.py`).**
  This cascade uses the OrthoDB group first, then a shared Pfam domain, then the UniRef50
  cluster. The first hit is kept, preferring reviewed entries, and the level that matched is
  recorded.
- **Reciprocal BLAST best hit:** not scripted in this repository (TODO, see README).

## 3. Pocket detection and descriptors

**Valid crystallographic ligand (`utils/ligand_characterization.py`).** A ligand is valid if it
meets all of the following:

- it is a HETATM residue that is not a standard amino acid, a common modification, water, an
  ion or a crystallization additive;
- it has more than 6 carbon atoms;
- its solvent accessibility (SASA in complex / SASA isolated) is below 0.4;
- its buriedness is above 15 (protein heavy atoms within 4.5 Å of any ligand heavy atom).

For each ligand the module also records:

- the residues within 8 Å of the ligand centroid;
- the fpocket pocket nearest to the ligand: its volume, score, druggability score, centroid
  distance and residue Jaccard overlap within 5 Å;
- the residue composition within 5 Å of the ligand. Residues are counted as hydrophobic (A, V,
  L, I, M, F, W, P, G), polar (S, T, N, Q, C, Y, H), charged (K, R, H, D, E) and aromatic
  (F, Y, W, H).

**Pocket centroid used for PocketVec (`utils/pipeline_steps.py`).**

1. If a ligand passes all filters, the centroid of the best-ranked ligand is used. When that
   ligand appears in several of the listed `ligand_chains`, the coordinates of all copies are
   combined.
2. Otherwise, which is always the case for AlphaFold models, fpocket is run on the prepared
   receptor and its pockets are rescored with P2Rank. The centroid of the pocket with the
   highest fpocket druggability score is used.

**Receptor preparation.** All ligands and waters are removed (the list is detected
automatically when `ligands_to_remove` is empty). Hydrogens are removed and alternate
locations are resolved by occupancy with `check_structure`. The receptor is then converted to
MOL2 with MOE if it is available, otherwise with Open Babel.

## 4. PocketVec fingerprints

- `rbcavity` uses the reference-ligand site mapper around the pocket centroid: radius 12 Å,
  small sphere 1.0 Å, minimum volume 100 Å³, one cavity.
- `rbdock` docks the 128 lead-like molecules of `TOP_128_rDock_LLM.sdf` as rigid ligands
  (`DIHEDRAL_MODE FIXED`), with 25 runs per molecule and seed 42.
- For each molecule, the minimum `SCORE.INTER` is kept. Scores are ordered as in the library
  order file and converted to ranks (`scipy.stats.rankdata`, `method="max"`). Positive scores
  are moved below all others: ranks n+1, n+2 and n+3 for scores in (0, 50), (50, 100) and
  > 100.

## 5. Pocket comparison and QC

- **Cosine distance.** Each pair's human and *T. cruzi* fingerprints are compared with
  `scipy.spatial.distance.cosine`.
- **Selectivity bands (`06_analyze_selectivity.py`).** Distances below 0.17 are classed as
  conserved (off-target risk), 0.17 to 0.20 as gray zone, and 0.20 or above as selective.
  Both thresholds can be changed from the command line. If at least three fingerprints are
  available, the script also draws a full N×N heatmap of cross-distances.
- **AlphaFold pocket QC (`07_qc_plddt_pocket.py`).** This check uses the per-residue pLDDT
  (AlphaFold DB JSON, or the PDB B-factor as fallback) and the PAE of the residues in the
  selected fpocket pocket. A model gets a red flag for each of these conditions:
  - pocket pLDDT < 70;
  - global pLDDT minus pocket pLDDT > 15;
  - mean intra-pocket PAE > 5 Å.

  It is classed as OK with no flags, REVISAR with exactly one flag (other than low pocket
  pLDDT), and DESCARTAR otherwise.

## Case study: MAGL (`src/case_studies/magl/`)

- **`superpose_magl.py`** identifies the catalytic triad on the *T. cruzi* side. It finds the
  GXSXG nucleophile-elbow motif, anchored on the pairwise-aligned human Ser, then the His
  closest to Ser-OG and the Asp/Glu closest to His-ND1. It then superposes the two structures
  with PyMOL `cealign`, reports the global RMSD and the triad RMSD, and refines locally when
  the triad RMSD is poor.
- **`subregions_magl.py`** compares the pocket residues of the two structures, checks that the
  triad is inside each pocket, and examines the oxyanion hole.
- **`tunnel_magl.py`** approximates the substrate tunnel with a geometric widest-path search
  (Dijkstra over a grid of free radii). This is an approximation used instead of CAVER, not a
  replacement for it.
- **`render_magl_views.py`** renders images from the saved PyMOL session.
