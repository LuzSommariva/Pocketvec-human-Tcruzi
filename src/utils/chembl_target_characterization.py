"""
ChEMBL target ID -> UniProt -> PDB humano caracterizado -> veredicto homologo T. cruzi.

Cinco fases, cada una filtra a la siguiente:
  Fase 1: ChEMBL target_id (local, chembl_37.db) -> UniProt accession(es) humano(s).
  Fase 2: UniProt humano -> homologo en T. cruzi (UniProt UniRef cluster + AlphaFold DB API).
          Filtra ANTES de tocar PDBs: not_found sale del pipeline.
  Fase 3: UniProt humano -> mejor PDB (PDBe best_structures API), marca si tiene ligando.
  Fase 4: Descarga + characterize_holo_pdb_ligand_pockets() (holo) o fpocket+p2rank sobre
          toda la proteina (fallback apo, sin ligando cristalizado).
  Fase 5: Consolida en un CSV esquema targets.csv + un CSV de descartes con motivo por fase.

No reimplementa characterize_holo_pdb_ligand_pockets() -- se importa tal cual de
ligand_characterization.py y se usa sin modificar su logica interna.
"""

import json
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# src/ (padre de utils/) tiene que estar en sys.path para poder importar
# utils.* como paquete cuando este modulo se corre directamente.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ligand_characterization import characterize_holo_pdb_ligand_pockets
from utils import config

# --------------------------------------------------------------------------
# Config -- paths and tool locations come from config/config.yaml.
# --------------------------------------------------------------------------
CHEMBL_DB = Path(config.CHEMBL_DB)

FPOCKET_BIN = config.PATH_TO_FPOCKET
PRANK_BIN = config.PATH_TO_PRANK
OBABEL_BIN = config.OBABEL_BIN

OUT_ROOT = Path(config.CHEMBL_TARGET_RUN_DIR)
STRUCT_DIR = OUT_ROOT / "structures"
LOG_DIR = OUT_ROOT / "logs"
for d in (OUT_ROOT, STRUCT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CRUZI_TAXID = 5693
HUMAN_TAXID = 9606

UNIPROT_REST = "https://rest.uniprot.org"
PDBE_API = "https://www.ebi.ac.uk/pdbe/api"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pocketvec-human-tcruzi"})

TARGETS_CSV_SCHEMA = [
    "target_id", "target_name", "human_pdb_code", "human_chain", "human_uniprot",
    "ligand_chains", "cruzi_uniprot", "cruzi_af_code", "cruzi_af_status", "cruzi_name",
]

DISCARD_COLUMNS = ["target_id", "target_name", "human_uniprot", "phase", "reason", "evidence"]


def _get_json(url, params=None, timeout=20, retries=3, backoff=1.5):
    last_exc = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    warnings.warn(f"GET {url} fallo tras {retries} intentos: {last_exc}")
    return None


# --------------------------------------------------------------------------
# Fase 1: ChEMBL target_id -> UniProt accession(es)
# --------------------------------------------------------------------------
def chembl_targets_to_uniprot(chembl_target_ids, chembl_db=CHEMBL_DB):
    """
    Consulta local a chembl_37.db (esquema estandar de ChEMBL):
      target_dictionary.chembl_id == chembl_target_id
      target_dictionary.tid -> target_components.tid -> component_id
      component_sequences.component_id -> accession (UniProt), organism, description

    Un target puede tener >1 componente (ej. complejos, homologue=1) -> se devuelve
    una fila por (target_id, accession); el llamador decide si toma solo el
    componente principal (homologue=0) o todos.
    """
    con = sqlite3.connect(str(chembl_db))
    placeholders = ",".join("?" for _ in chembl_target_ids)
    query = f"""
        SELECT
            td.chembl_id   AS target_id,
            td.pref_name   AS target_name,
            td.tid         AS tid,
            td.target_type AS target_type,
            td.organism    AS target_organism,
            tc.homologue   AS homologue,
            cs.accession   AS uniprot_accession,
            cs.component_type AS component_type,
            cs.organism    AS component_organism,
            cs.description AS component_description
        FROM target_dictionary td
        JOIN target_components tc ON tc.tid = td.tid
        JOIN component_sequences cs ON cs.component_id = tc.component_id
        WHERE td.chembl_id IN ({placeholders})
    """
    df = pd.read_sql_query(query, con, params=list(chembl_target_ids))
    con.close()
    return df


# --------------------------------------------------------------------------
# Fase 2: homologo en T. cruzi via UniProt (UniRef) + AlphaFold DB
# --------------------------------------------------------------------------
def _uniref_cluster_for_accession(accession, identity=0.9):
    """
    Busca el cluster UniRef al que pertenece un accession, al nivel de identidad dado
    (0.9 = UniRef90, 0.5 = UniRef50). OJO: 'identity' tiene que ir DENTRO del string de
    'query' (query=uniprot_id:ACC AND identity:0.9) -- pasarlo como parametro de URL
    aparte no tiene efecto y siempre devuelve el cluster UniRef100.
    """
    data = _get_json(
        f"{UNIPROT_REST}/uniref/search",
        params={"query": f"uniprot_id:{accession} AND identity:{identity}", "fields": "id,name", "format": "json", "size": 1},
    )
    if not data or not data.get("results"):
        return None
    return data["results"][0]["id"]


def _uniref_cluster_members(cluster_id):
    data = _get_json(f"{UNIPROT_REST}/uniref/{cluster_id}", params={"fields": "id,name,count,members"})
    return data


def _find_cruzi_member_in_cluster(cluster_json):
    """Recorre representative member + members buscando taxonID 5693 (T. cruzi)."""
    if not cluster_json:
        return None
    candidates = []
    rep = cluster_json.get("representativeMember")
    if rep:
        candidates.append(rep)
    candidates.extend(cluster_json.get("members", []) or [])

    for m in candidates:
        organism = m.get("organismTaxId") or (m.get("organism") or {}).get("taxonId")
        if organism == CRUZI_TAXID:
            acc = m.get("accession") or m.get("accessions", [None])[0]
            name = m.get("proteinName") or (m.get("organism") or {}).get("scientificName")
            return {"cruzi_uniprot": acc, "cruzi_name": name}
    return None


def _alphafold_status(accession):
    """found: hay modelo AF publicado. not_found: AF no tiene ni el accession existe."""
    if not accession:
        return "not_found", None
    data = _get_json(f"{ALPHAFOLD_API}/{accession}")
    if data and isinstance(data, list) and len(data) > 0:
        af_code = data[0].get("entryId") or f"AF-{accession}-F1"
        return "found", af_code
    return "needs_prediction", None


def _uniprot_orthodb_ids(accession):
    data = _get_json(f"{UNIPROT_REST}/uniprotkb/{accession}", params={"fields": "xref_orthodb"})
    if not data:
        return []
    return [x["id"] for x in data.get("uniProtKBCrossReferences", []) if x.get("database") == "OrthoDB"]


def _cruzi_entries_for_orthodb(group_id, size=10):
    data = _get_json(
        f"{UNIPROT_REST}/uniprotkb/search",
        params={
            "query": f"(taxonomy_id:{CRUZI_TAXID}) (xref:orthodb-{group_id})",
            "fields": "accession,protein_name,organism_name,reviewed",
            "format": "json",
            "size": size,
        },
    )
    if not data:
        return []
    return data.get("results", [])


def _uniprot_pfam_ids(accession):
    data = _get_json(f"{UNIPROT_REST}/uniprotkb/{accession}", params={"fields": "xref_pfam"})
    if not data:
        return []
    return [x["id"] for x in data.get("uniProtKBCrossReferences", []) if x.get("database") == "Pfam"]


def _cruzi_entries_for_pfam(pfam_id, size=10):
    """
    Entradas T. cruzi (taxid 5693) que comparten un dominio Pfam con el humano.
    Prefiere reviewed (Swiss-Prot) sobre TrEMBL -- se ordena en el llamador.
    NOTA: 'taxonomy_id:5693 AND xref:pfam-PFxxxx' devuelve 0 resultados (bug/quirk
    del parser de busqueda de UniProt que garabatea el numero de taxonomia cuando
    se combina con AND); en cambio "(taxonomy_id:5693) (xref:pfam-PFxxxx)" (AND
    implicito por espacio) funciona bien -- confirmado empiricamente.
    """
    data = _get_json(
        f"{UNIPROT_REST}/uniprotkb/search",
        params={
            "query": f"(taxonomy_id:{CRUZI_TAXID}) (xref:pfam-{pfam_id})",
            "fields": "accession,protein_name,organism_name,reviewed",
            "format": "json",
            "size": size,
        },
    )
    if not data:
        return []
    return data.get("results", [])


def find_cruzi_homolog(human_uniprot):
    """
    Devuelve dict con veredicto sobre homologo en T. cruzi para un UniProt humano.
    Cascada de 3 criterios, todos con UniProt como unica fuente, de mas a menos
    especifico:

    1) OrthoDB (grupo de ortologia calculado filogeneticamente, cross-ref en la
       entrada UniProt). Es el criterio PREFERIDO: en la tanda de validacion,
       el criterio por Pfam (ver 2) le asigno a KCNH2 (canal de K+, sin dominio
       quinasa) el mismo "homologo" generico ("non-specific serine/threonine
       protein kinase") que a EGFR y JAK2 -- via el dominio PF00027 (cNMP-binding),
       compartido tambien por subunidades reguladoras de quinasas. Con OrthoDB el
       mismo KCNH2 matchea a una "Cyclic nucleotide-binding domain-containing
       protein" de T. cruzi, mucho mas plausible. OrthoDB usa arboles filogeneticos,
       no solo co-ocurrencia de dominio, así que es mucho menos propenso a este
       tipo de falso positivo por dominio compartido pero funcionalmente distinto.
    2) Pfam (dominio compartido) -- fallback si la entrada humana no tiene grupo
       OrthoDB anotado. Mejor que similitud de secuencia cruda para esta distancia
       evolutiva (ver nota abajo) pero mas propenso a falsos positivos por dominios
       reguladores/de union muy difundidos (cNMP-binding, dominios PAS, etc.) --
       usar con cautela, revisar manualmente antes de confiar en runs grandes.
    3) UniRef50 (identidad de secuencia global, 50%) -- ultimo recurso. Documentado
       como insuficiente en soledad: GAPDH humano/T. cruzi (ortologos bien conocidos,
       EC 1.2.1.12) NO comparten cluster UniRef50 pese a compartir dominio Pfam
       PF00044/PF02800 y grupo OrthoDB -- T. cruzi es evolutivamente muy distante
       (Discoba) y la identidad de secuencia global subestima homologia remota.

    Se prueba en ese orden y se usa el primer hit; el nivel usado queda en
    evidence["matched_at"] para poder auditar cada caso.
    """
    evidence = {"human_uniprot": human_uniprot}

    orthodb_ids = _uniprot_orthodb_ids(human_uniprot)
    evidence["human_orthodb_ids"] = orthodb_ids

    for group_id in orthodb_ids:
        hits = _cruzi_entries_for_orthodb(group_id)
        evidence.setdefault("orthodb_hits", {})[group_id] = [h["primaryAccession"] for h in hits]
        if not hits:
            continue
        hits_sorted = sorted(hits, key=lambda h: h.get("entryType", "").startswith("UniProtKB reviewed"), reverse=True)
        best = hits_sorted[0]
        cruzi_acc = best["primaryAccession"]
        cruzi_name = best.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
        af_status, af_code = _alphafold_status(cruzi_acc)
        evidence["matched_at"] = f"orthodb:{group_id}"
        return {
            "cruzi_uniprot": cruzi_acc,
            "cruzi_name": cruzi_name,
            "cruzi_af_status": af_status,
            "cruzi_af_code": af_code,
            "cruzi_homolog_status": "found",
            "evidence": json.dumps(evidence, ensure_ascii=False),
        }

    pfam_ids = _uniprot_pfam_ids(human_uniprot)
    evidence["human_pfam_ids"] = pfam_ids

    for pfam_id in pfam_ids:
        hits = _cruzi_entries_for_pfam(pfam_id)
        evidence.setdefault("pfam_hits", {})[pfam_id] = [h["primaryAccession"] for h in hits]
        if not hits:
            continue
        hits_sorted = sorted(hits, key=lambda h: h.get("entryType", "").startswith("UniProtKB reviewed"), reverse=True)
        best = hits_sorted[0]
        cruzi_acc = best["primaryAccession"]
        cruzi_name = best.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
        af_status, af_code = _alphafold_status(cruzi_acc)
        evidence["matched_at"] = f"pfam:{pfam_id}"
        return {
            "cruzi_uniprot": cruzi_acc,
            "cruzi_name": cruzi_name,
            "cruzi_af_status": af_status,
            "cruzi_af_code": af_code,
            "cruzi_homolog_status": "found",
            "evidence": json.dumps(evidence, ensure_ascii=False),
        }

    # Fallback: sin Pfam anotado en la entrada humana -- reintentar con UniRef50.
    cluster_id = _uniref_cluster_for_accession(human_uniprot, identity=0.5)
    evidence["uniref50_cluster_fallback"] = cluster_id
    if cluster_id:
        cluster_json = _uniref_cluster_members(cluster_id)
        hit = _find_cruzi_member_in_cluster(cluster_json)
        if hit:
            af_status, af_code = _alphafold_status(hit["cruzi_uniprot"])
            evidence["matched_at"] = "uniref50_fallback"
            return {
                "cruzi_uniprot": hit["cruzi_uniprot"],
                "cruzi_name": hit["cruzi_name"],
                "cruzi_af_status": af_status,
                "cruzi_af_code": af_code,
                "cruzi_homolog_status": "found",
                "evidence": json.dumps(evidence, ensure_ascii=False),
            }

    return {
        "cruzi_uniprot": None,
        "cruzi_name": None,
        "cruzi_af_status": "not_found",
        "cruzi_af_code": None,
        "cruzi_homolog_status": "not_found",
        "evidence": json.dumps(evidence, ensure_ascii=False),
    }


# --------------------------------------------------------------------------
# Fase 3: UniProt humano -> mejor PDB (PDBe best_structures, ya rankeado)
# --------------------------------------------------------------------------
def best_pdb_candidates(human_uniprot, max_candidates=8):
    """
    OJO Fase 0: el endpoint viejo /mappings/uniprot/{acc} (SIFTS crudo) da 404
    ("Requested endpoint does not contain any data") -- parece deprecado/caido,
    probado incluso con targets con estructura conocida (EGFR, hemoglobina).
    Se usa en su lugar /mappings/best_structures/{acc}, que sigue vivo y ya
    viene rankeado por resolucion/cobertura -- mejor punto de partida.
    """
    data = _get_json(f"{PDBE_API}/mappings/best_structures/{human_uniprot}")
    if not data or human_uniprot not in data:
        return []
    return data[human_uniprot][:max_candidates]


def pdb_entry_ligand_count(pdb_id):
    """number_of_entities.ligand del summary de PDBe; >0 sugiere HETATM no trivial.
    Es solo un pre-filtro barato -- la identidad/validez real del ligando la decide
    characterize_holo_pdb_ligand_pockets() sobre la estructura descargada."""
    data = _get_json(f"{PDBE_API}/pdb/entry/summary/{pdb_id.lower()}")
    if not data or pdb_id.lower() not in data:
        return None
    entry = data[pdb_id.lower()][0]
    return entry.get("number_of_entities", {}).get("ligand")


def choose_best_pdb(human_uniprot, max_checked=6):
    """
    Preferencia (a) tiene ligando, (b) buena resolucion -- best_structures ya
    viene ordenado por resolucion/cobertura, asi que solo hace falta encontrar
    el primero de la lista con ligand_count > 0.
    Devuelve (chosen_dict, all_candidates, has_any_ligand_pdb).
    """
    candidates = best_pdb_candidates(human_uniprot, max_candidates=max_checked)
    if not candidates:
        return None, [], False

    seen_pdb = []
    for c in candidates:
        pdb_id = c["pdb_id"].upper()
        if pdb_id in [s["pdb_id"] for s in seen_pdb]:
            continue
        lig_count = pdb_entry_ligand_count(pdb_id)
        c = dict(c, pdb_id=pdb_id, ligand_count=lig_count)
        seen_pdb.append(c)
        if lig_count and lig_count > 0:
            return c, seen_pdb, True

    # Ninguno con ligando: se usa el de mejor resolucion (primero de la lista) para el fallback apo.
    return seen_pdb[0], seen_pdb, False


# --------------------------------------------------------------------------
# Fase 4: descarga + caracterizacion
# --------------------------------------------------------------------------
def fetch_rcsb_structure(pdb_code, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{pdb_code}.pdb"
    if outfile.exists() and outfile.stat().st_size > 0:
        return outfile
    r = SESSION.get(f"{RCSB_DOWNLOAD}/{pdb_code}.pdb", timeout=30)
    if r.status_code != 200 or not r.text.strip():
        return None
    outfile.write_text(r.text)
    return outfile


def characterize_holo(pdb_code, workdir, return_all=False):
    """Wrapper delgado sobre characterize_holo_pdb_ligand_pockets(), sin tocar su logica.
    Solo fija path_to_fpocket al binario configurado."""
    df = characterize_holo_pdb_ligand_pockets(
        pdb_code,
        outpath=str(workdir),
        path_to_fpocket=FPOCKET_BIN,
        run_fpocket=True,
        overwrite_fpocket=False,
        return_all=return_all,
        verbose=False,
    )
    return df


def run_fpocket_prank_apo(pdb_path, workdir):
    """
    Fallback sin ligando cristalizado, mismo patron que el pipeline principal usa
    para T. cruzi sin ligando: fpocket + P2RANK sobre toda la proteina, elige el
    pocket de mayor fpocket_druggability_score.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pdb_path = Path(pdb_path)

    # --- fpocket ---
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        os.system(f"{FPOCKET_BIN} -f {pdb_path.name}" if pdb_path.parent == workdir else f"{FPOCKET_BIN} -f {pdb_path}")
    finally:
        os.chdir(cwd)

    stem = pdb_path.stem
    info_path = workdir / f"{stem}_out" / f"{stem}_info.txt"
    fpocket_pockets = {}
    if info_path.exists():
        from utils.ligand_characterization import _fallback_read_fpocket_scores
        fpocket_pockets = _fallback_read_fpocket_scores(info_path)

    best_pocket_id, best_drug_score = None, -np.inf
    for pid, d in fpocket_pockets.items():
        score = d.get("Druggability Score", d.get("Drug Score", np.nan))
        if pd.notna(score) and score > best_drug_score:
            best_pocket_id, best_drug_score = pid, score

    # --- p2rank (informativo / cross-check, no reemplaza la eleccion por fpocket) ---
    prank_outdir = workdir / "p2rank_out"
    os.system(f"{PRANK_BIN} predict -f {pdb_path} -o {prank_outdir} > {workdir / 'p2rank.log'} 2>&1")
    prank_csv = prank_outdir / f"{pdb_path.name}_predictions.csv"
    prank_top_score = None
    if prank_csv.exists():
        try:
            pr = pd.read_csv(prank_csv)
            pr.columns = [c.strip() for c in pr.columns]
            if len(pr) > 0:
                prank_top_score = float(pr.sort_values("score", ascending=False).iloc[0]["score"])
        except Exception as e:
            warnings.warn(f"No pude leer p2rank csv {prank_csv}: {e}")

    return {
        "fpocket_best_pocket_id": best_pocket_id,
        "fpocket_druggability_score": best_drug_score if best_pocket_id is not None else np.nan,
        "fpocket_all_pockets": json.dumps(fpocket_pockets, default=str),
        "p2rank_top_pocket_score": prank_top_score,
        "fpocket_outdir": str(workdir / f"{stem}_out"),
        "p2rank_outdir": str(prank_outdir),
    }


# --------------------------------------------------------------------------
# Driver: corre las 5 fases sobre una lista de ChEMBL target IDs
# --------------------------------------------------------------------------
def run_pipeline(chembl_target_ids, prefer_homologue0=True):
    consolidated_rows = []
    discard_rows = []

    print(f"=== Fase 1: {len(chembl_target_ids)} ChEMBL target IDs -> UniProt ===")
    map_df = chembl_targets_to_uniprot(chembl_target_ids)
    if map_df.empty:
        print("Fase 1 no devolvio nada -- revisar chembl_target_ids / DB.")
        return pd.DataFrame(columns=TARGETS_CSV_SCHEMA), pd.DataFrame(columns=DISCARD_COLUMNS)

    found_ids = set(map_df["target_id"])
    for tid in chembl_target_ids:
        if tid not in found_ids:
            discard_rows.append({"target_id": tid, "target_name": None, "human_uniprot": None,
                                  "phase": "1_chembl_to_uniprot", "reason": "target_id no encontrado en target_dictionary",
                                  "evidence": ""})

    # component_type == 'PROTEIN' y organism humano; homologue=0 es el componente "principal".
    human_map = map_df[(map_df["component_organism"] == "Homo sapiens") & (map_df["component_type"] == "PROTEIN")].copy()
    if prefer_homologue0 and (human_map["homologue"] == 0).any():
        human_map = human_map[human_map["homologue"] == 0]

    no_human_component = set(map_df["target_id"]) - set(human_map["target_id"])
    for tid in no_human_component:
        row = map_df[map_df["target_id"] == tid].iloc[0]
        discard_rows.append({"target_id": tid, "target_name": row["target_name"], "human_uniprot": None,
                              "phase": "1_chembl_to_uniprot",
                              "reason": f"sin componente PROTEIN humano (organism target={row['target_organism']})",
                              "evidence": ""})

    print(f"Fase 1 OK: {human_map['target_id'].nunique()} targets con >=1 UniProt humano "
          f"({len(human_map)} filas -- multi-accession donde aplica)")

    print("\n=== Fase 2: filtro por homologo T. cruzi (antes de tocar PDBs) ===")
    phase2_pass = []
    for _, row in human_map.iterrows():
        tid, tname, acc = row["target_id"], row["target_name"], row["uniprot_accession"]
        verdict = find_cruzi_homolog(acc)
        print(f"  {tid} ({tname}) UniProt={acc} -> cruzi={verdict['cruzi_homolog_status']} "
              f"(af_status={verdict['cruzi_af_status']}, cruzi_uniprot={verdict['cruzi_uniprot']})")
        if verdict["cruzi_homolog_status"] == "not_found":
            discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                  "phase": "2_cruzi_homolog", "reason": "sin homologo T. cruzi (UniRef90/50, AlphaFold DB)",
                                  "evidence": verdict["evidence"]})
            continue
        phase2_pass.append({**row.to_dict(), **verdict})

    phase2_df = pd.DataFrame(phase2_pass)
    print(f"Fase 2 OK: {len(phase2_df)}/{len(human_map)} filas pasan (con homologo cruzi found/needs_prediction)")

    if phase2_df.empty:
        return pd.DataFrame(columns=TARGETS_CSV_SCHEMA), pd.DataFrame(discard_rows, columns=DISCARD_COLUMNS)

    print("\n=== Fase 3: elegir mejor PDB humano por UniProt ===")
    phase3_rows = []
    for _, row in phase2_df.iterrows():
        tid, tname, acc = row["target_id"], row["target_name"], row["uniprot_accession"]
        chosen, candidates, has_ligand = choose_best_pdb(acc)
        if chosen is None:
            discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                  "phase": "3_choose_pdb", "reason": "PDBe best_structures sin resultados para este UniProt",
                                  "evidence": ""})
            continue
        print(f"  {tid} UniProt={acc} -> {chosen['pdb_id']} chain={chosen.get('chain_id')} "
              f"res={chosen.get('resolution')} ligand_count={chosen.get('ligand_count')} "
              f"has_any_ligand_pdb={has_ligand} (evaluados: {[c['pdb_id'] for c in candidates]})")
        phase3_rows.append({**row.to_dict(), "chosen_pdb": chosen, "has_ligand_pdb": has_ligand,
                             "candidates_evaluated": [c["pdb_id"] for c in candidates]})

    print(f"Fase 3 OK: {len(phase3_rows)} targets con PDB elegido "
          f"({sum(r['has_ligand_pdb'] for r in phase3_rows)} con ligando, "
          f"{sum(not r['has_ligand_pdb'] for r in phase3_rows)} van a fallback apo)")

    print("\n=== Fase 4: descarga + caracterizacion ===")
    for r in phase3_rows:
        tid, tname, acc = r["target_id"], r["target_name"], r["uniprot_accession"]
        pdb_id = r["chosen_pdb"]["pdb_id"]
        workdir = STRUCT_DIR / tid
        pdb_path = fetch_rcsb_structure(pdb_id, workdir)

        if pdb_path is None:
            discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                  "phase": "4_download", "reason": f"no pude descargar {pdb_id} de RCSB",
                                  "evidence": ""})
            continue

        base_row = {
            "target_id": tid, "target_name": tname, "human_uniprot": acc,
            "cruzi_uniprot": r["cruzi_uniprot"], "cruzi_af_code": r["cruzi_af_code"],
            "cruzi_af_status": r["cruzi_af_status"], "cruzi_name": r["cruzi_name"],
        }

        if r["has_ligand_pdb"]:
            try:
                char_df = characterize_holo(pdb_id, workdir)
            except Exception as e:
                discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                      "phase": "4_characterize_holo", "reason": f"excepcion en characterize_holo_pdb_ligand_pockets: {e}",
                                      "evidence": ""})
                continue

            if char_df.empty:
                discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                      "phase": "4_characterize_holo",
                                      "reason": f"{pdb_id} listado con ligando por PDBe pero ningun HETATM paso los filtros de characterize_holo_pdb_ligand_pockets",
                                      "evidence": ""})
                continue

            best = char_df.iloc[0].to_dict()
            row = {
                **base_row,
                "human_pdb_code": pdb_id,
                "human_chain": best["chain"],
                "ligand_chains": best["chain"],
                "route": "holo_ligand",
                **best,
            }
            consolidated_rows.append(row)
            print(f"  {tid}: {pdb_id} holo OK -> ligando {best['ligand_id']} ({best['docking_suitability']})")

        else:
            try:
                apo_metrics = run_fpocket_prank_apo(pdb_path, workdir)
            except Exception as e:
                discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                      "phase": "4_characterize_apo", "reason": f"excepcion en fallback fpocket/p2rank: {e}",
                                      "evidence": ""})
                continue

            if apo_metrics["fpocket_best_pocket_id"] is None:
                discard_rows.append({"target_id": tid, "target_name": tname, "human_uniprot": acc,
                                      "phase": "4_characterize_apo", "reason": f"fpocket no encontro pockets en {pdb_id} (apo)",
                                      "evidence": ""})
                continue

            row = {
                **base_row,
                "human_pdb_code": pdb_id,
                "human_chain": r["chosen_pdb"].get("chain_id"),
                "ligand_chains": "",
                "route": "apo_fpocket_p2rank_fallback",
                **apo_metrics,
            }
            consolidated_rows.append(row)
            print(f"  {tid}: {pdb_id} apo fallback OK -> pocket {apo_metrics['fpocket_best_pocket_id']} "
                  f"(druggability={apo_metrics['fpocket_druggability_score']:.3f})")

    print("\n=== Fase 5: consolidar ===")
    consolidated_df = pd.DataFrame(consolidated_rows)
    discards_df = pd.DataFrame(discard_rows, columns=DISCARD_COLUMNS)

    if not consolidated_df.empty:
        for col in TARGETS_CSV_SCHEMA:
            if col not in consolidated_df.columns:
                consolidated_df[col] = np.nan
        ordered_cols = TARGETS_CSV_SCHEMA + [c for c in consolidated_df.columns if c not in TARGETS_CSV_SCHEMA]
        consolidated_df = consolidated_df[ordered_cols]

    return consolidated_df, discards_df


if __name__ == "__main__":
    import sys

    validation_batch = [
        "CHEMBL203",      # EGFR - kinase, muchos PDB con ligando
        "CHEMBL2971",     # JAK2 - kinase, muchos PDB con ligando
        "CHEMBL205",      # Carbonic anhydrase 2 - muchos PDB con ligando
        "CHEMBL220",      # Acetylcholinesterase - PDB con ligando
        "CHEMBL251",      # Adenosine A2a - GPCR con ligandos
        "CHEMBL325",      # HDAC1 - enzima con ligandos
        "CHEMBL240",      # hERG/KCNH2 - canal ionico, candidato a fallback apo
        "CHEMBL4296",     # Nav1.7/SCN9A - canal ionico, candidato a fallback apo
        "CHEMBL233",      # Mu-opioid receptor - GPCR
        "CHEMBL1293224",  # MAPT/Tau - caso limite (fibrillas, HETATM atipicos)
    ]

    batch = sys.argv[1:] if len(sys.argv) > 1 else validation_batch

    consolidated, discards = run_pipeline(batch)

    consolidated_path = OUT_ROOT / "targets_chembl_consolidated.csv"
    discards_path = OUT_ROOT / "targets_chembl_discards.csv"
    consolidated.to_csv(consolidated_path, index=False)
    discards.to_csv(discards_path, index=False)

    print(f"\nConsolidado: {len(consolidated)} filas -> {consolidated_path}")
    print(f"Descartes:   {len(discards)} filas -> {discards_path}")
