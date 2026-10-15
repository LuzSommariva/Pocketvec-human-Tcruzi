"""
Filtro de familia PRE-BLAST para Fase 3 (ver PIPELINE_REPORT.md seccion 9).
Excluye candidatos GPCR clase A / receptor nuclear de hormonas ANTES de gastar
BLAST en ellos -- estas dos familias explicaron ~30/33 de los not_found en los
2 batches anteriores, sin que la posicion en el ranking lo prediga.

Clasificacion por DOMINIO/FAMILIA real (UniProt), nunca por nombre de proteina:
  - GPCR clase A: Pfam PF00001 (7tm_1) O comentario SIMILARITY de UniProt
    "G protein-coupled receptor 1 family" (misma familia Pfam/UniProt que ya
    se usa para el chequeo de arquitectura de dominios en Fase 3, no un metodo
    nuevo).
  - Receptor nuclear de hormonas: Pfam PF00104 (Hormone_recep, dominio de
    union a ligando) y/o PF00105 (zf-C4, dominio de union a DNA) O comentario
    SIMILARITY "nuclear hormone receptor family".
"""

import requests
import time

UNIPROT_REST = "https://rest.uniprot.org"

GPCR_PFAM = {"PF00001"}
NUCLEAR_RECEPTOR_PFAM = {"PF00104", "PF00105"}


def _get_json(url, params=None, timeout=20, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def classify_family(accession):
    """
    Devuelve dict: {"pfam": [...], "similarity": "...", "excluded": bool,
    "exclusion_reason": str|None}
    """
    data = _get_json(f"{UNIPROT_REST}/uniprotkb/{accession}",
                      params={"fields": "xref_pfam,cc_similarity"})
    if not data:
        return {"pfam": [], "similarity": "", "excluded": False,
                "exclusion_reason": None, "note": "no se pudo consultar UniProt"}

    pfam = [x["id"] for x in data.get("uniProtKBCrossReferences", []) if x.get("database") == "Pfam"]
    similarity = ""
    for c in data.get("comments", []):
        if c.get("commentType") == "SIMILARITY":
            texts = c.get("texts", [])
            if texts:
                similarity = texts[0].get("value", "")

    pfam_set = set(pfam)
    is_gpcr = bool(pfam_set & GPCR_PFAM) or "G protein-coupled receptor 1 family" in similarity \
        or "G-protein coupled receptor" in similarity
    is_nuclear_receptor = bool(pfam_set & NUCLEAR_RECEPTOR_PFAM) or "nuclear hormone receptor family" in similarity

    excluded = is_gpcr or is_nuclear_receptor
    reason = None
    if is_gpcr:
        reason = f"GPCR clase A (Pfam={sorted(pfam_set & GPCR_PFAM)}, similarity='{similarity}')"
    elif is_nuclear_receptor:
        reason = f"Receptor nuclear de hormonas (Pfam={sorted(pfam_set & NUCLEAR_RECEPTOR_PFAM)}, similarity='{similarity}')"

    return {"pfam": pfam, "similarity": similarity, "excluded": excluded, "exclusion_reason": reason}
