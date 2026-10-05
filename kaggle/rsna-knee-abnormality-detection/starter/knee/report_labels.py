"""Radiology report -> 12 soft training targets.

Only 58 of 4,407 training studies carry radiologist labels; every study carries its report
(9 languages). Test studies have NO report, so the text is a training-time teacher only.

Every reader here (LLM or regex) emits one *status* per finding; `statuses_to_soft` turns
statuses into soft targets + per-cell loss weights. Two forum findings drive the design:
  * Ask for "not mentioned" explicitly (~25% of cells). What silence means differs by
    finding: a report silent on Baker's cyst almost never has one (gold+ 3%), but silent
    synovitis is still positive ~34% of the time (it is rarely written down). So silence
    gets a per-finding prior, and silent synovitis is imputed from effusion (they co-occur).
  * The 58 gold labels come from image review with STRICTER thresholds than clinical
    reports (moderate/large effusion, high-grade ligament tear, >50% cartilage loss), so a
    "small effusion" or "low-grade sprain" is mostly negative -> a "below_threshold" status.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .config import LABELS

KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa",
        "pf_oa", "effusion", "synovitis", "bakers_cyst", "contusion", "fracture"]
KEY2LABEL = dict(zip(KEYS, LABELS))

STATUSES = ["positive", "probable", "equivocal", "below_threshold", "negative", "not_mentioned"]
STATUS_VALUE = {"positive": 0.92, "probable": 0.75, "equivocal": 0.5,
                "below_threshold": 0.25, "negative": 0.04}
# P(radiologist label = 1 | report silent): rough defaults, tune on the 58 gold + LB
SILENCE_PRIOR = {"ACL": 0.03, "MCL": 0.03, "Medial Meniscus": 0.08, "Lateral Meniscus": 0.08,
                 "Medial OA": 0.05, "Lateral OA": 0.05, "PF OA": 0.15, "Effusion": 0.2,
                 "Synovitis": None, "Baker's": 0.03, "Contusion": 0.08, "Fracture": 0.1}
# silent cells for findings where silence is uninformative get a smaller loss weight
SILENCE_WEIGHT = {"PF OA": 0.6, "Effusion": 0.6, "Synovitis": 0.5, "Contusion": 0.8, "Fracture": 0.8}


def statuses_to_soft(status: pd.DataFrame) -> pd.DataFrame:
    """status: index StudyInstanceUID, columns LABELS with values in STATUSES.
    Returns soft targets (LABELS) and loss weights (w_<label>)."""
    out = pd.DataFrame(index=status.index)
    for lab in LABELS:
        s = status[lab].fillna("not_mentioned")
        v = s.map(STATUS_VALUE).astype(float)
        silent = s.eq("not_mentioned")
        if SILENCE_PRIOR[lab] is not None:
            v[silent] = SILENCE_PRIOR[lab]
        out[lab] = v
        out[f"w_{lab}"] = np.where(silent, SILENCE_WEIGHT.get(lab, 1.0), 1.0)
    # synovitis is under-reported but co-occurs with effusion: impute only the silent cells
    syn_silent = status["Synovitis"].fillna("not_mentioned").eq("not_mentioned")
    eff = out["Effusion"]
    out.loc[syn_silent, "Synovitis"] = np.select(
        [eff[syn_silent] >= 0.7, eff[syn_silent] >= 0.4, eff[syn_silent] >= 0.2],
        [0.6, 0.4, 0.3], 0.15)
    return out


def evaluate_against_gold(soft: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Per-label and macro AUC of a label table vs the 58 radiologist-labelled studies.
    58 studies, 9-35 positives per label: differences < ~0.02 macro are noise."""
    from sklearn.metrics import roc_auc_score
    gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
    common = gold.index.intersection(soft.index)
    res = {}
    for lab in LABELS:
        y = gold.loc[common, lab].astype(int)
        p = soft.loc[common, lab].astype(float)
        if 0 < y.sum() < len(y):
            res[lab] = roc_auc_score(y, p)
    s = pd.Series(res)
    s["macro"] = s.mean()
    s["n_gold"] = len(common)
    return s


# ------------------------------------------------------------------ LLM prompt + schema
RUBRIC = """You are a musculoskeletal radiologist grading knee MRI reports for a research dataset.
You will receive ONE radiology report. It may be written in English, Spanish, Turkish,
Croatian/Bosnian/Serbian, Greek, German, Bulgarian, Dutch, French or Italian. Read it in its
original language (do not translate it in your answer) and grade 12 findings for the examined
knee. If both knees are described, grade the knee that the report is mainly about.

The dataset's ground truth was made by MSK radiologists reading the IMAGES with strict
thresholds; borderline ("on the fence") findings were graded negative. Grade each finding
against these criteria:

- acl: high-grade partial (>50% of fibres) or complete tear of the anterior cruciate ligament
  (incl. torn/deficient ACL graft). Mild signal change, mucoid degeneration, thickening or a
  low-grade sprain without discontinuity is below threshold.
- mcl: high-grade partial or complete ACUTE tear of the medial collateral ligament (disrupted
  fibres with edema). Low-grade sprain, periligamentous edema only, or chronic/remote
  thickening (e.g. Pellegrini-Stieda) is below threshold.
- medial_meniscus: tear of the medial meniscus = abnormal signal reaching the articular surface
  (grade 3), or a morphologic abnormality (truncated, diminutive, displaced/flipped fragment,
  bucket-handle, root tear, complex/macerated). Intrasubstance degeneration that does not reach
  the surface (grade 1-2, "degenerative signal", "mucoid") is below threshold. A post-
  meniscectomy truncated meniscus counts as positive.
- lateral_meniscus: same criteria for the lateral meniscus.
- medial_oa: osteoarthritis of the MEDIAL tibiofemoral compartment = a moderate or large area
  (~1 cm or more) of high-grade cartilage loss (>50% thickness; grade 3-4, full-thickness,
  denuded/exposed bone, bone-on-bone), with or without osteophytes/subchondral changes. Mild or
  superficial chondropathy/chondromalacia (grade 1-2), fissuring or "early" OA is below threshold.
- lateral_oa: same for the LATERAL tibiofemoral compartment.
- pf_oa: same for the PATELLOFEMORAL compartment (patella and/or trochlea).
  "Tricompartmental" osteoarthritis applies to all three compartments.
- effusion: MODERATE or LARGE joint effusion. Small/minimal/trace/physiologic fluid is below
  threshold; an effusion with no size given is "probable".
- synovitis: inflammation/thickening of the synovial lining (synovitis, synovial hypertrophy or
  proliferation, effusion-synovitis, Hoffa-synovitis). Do not infer it from effusion alone.
- bakers_cyst: MODERATE or LARGE popliteal (Baker's) cyst, incl. ruptured cyst. A small cyst or
  a little fluid in the gastrocnemius-semimembranosus bursa is below threshold.
- contusion: bone contusion / bone bruise / traumatic bone-marrow edema WITHOUT a discrete
  fracture line. Degenerative subchondral edema (with OA), stress reaction, or edema next to
  cysts is below threshold.
- fracture: ACUTE fracture (cortical break or fracture line), incl. impaction, avulsion
  (e.g. Segond) and acute osteochondral fracture. Old/healed fractures and post-operative
  hardware are negative; subchondral insufficiency fracture is equivocal.

Status for each finding (choose exactly one):
- "positive": the report states the finding and it meets the criteria above.
- "probable": likely/suspected, or present but severity not stated where severity matters.
- "equivocal": mentioned in a way that leaves it genuinely uncertain.
- "below_threshold": present, but explicitly milder/smaller than the criteria above.
- "negative": explicitly absent, or the structure is described as normal/intact (a blanket
  statement such as "rest of the examination is normal" counts as negative).
- "not_mentioned": the report says nothing that lets you judge this finding.
"not_mentioned" is a valid, important answer: never guess from other findings.
"""

USER_TEMPLATE = "Radiology report:\n<report>\n{report}\n</report>"


def output_schema(with_evidence: bool = True) -> dict:
    item_props = {"status": {"type": "string", "enum": STATUSES}}
    if with_evidence:
        item_props["evidence"] = {"type": "string", "description":
                                  "<= 12 words quoted from the report, original language; empty if none"}
    item = {"type": "object", "properties": item_props,
            "required": list(item_props), "additionalProperties": False}
    return {"type": "object",
            "properties": {k: item for k in KEYS},
            "required": KEYS, "additionalProperties": False}


def parse_llm_json(obj: dict) -> dict:
    """LLM JSON -> {LABEL: status}; anything malformed becomes not_mentioned."""
    out = {}
    for k in KEYS:
        st = (obj.get(k) or {}).get("status") if isinstance(obj.get(k), dict) else obj.get(k)
        out[KEY2LABEL[k]] = st if st in STATUSES else "not_mentioned"
    return out


# ------------------------------------------------------------------ regex fallback
def normalize(text: str) -> str:
    t = str(text).replace("ı", "i").replace("İ", "i").replace("ß", "ss").replace("µ", "μ")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c)).replace("ς", "σ")
    return t


def clauses(text: str) -> list[str]:
    t = re.sub(r":\s*\n\s*", ": ", text)  # "Fractures :\nAucune." is ONE statement
    parts = re.split(r"[.;!?\n]+|\b(?:but|however|pero|sin embargo|aber|jedoch|mais|cependant|maar|echter"
                     r"|ancak|fakat|ali|medutim|αλλα|ομωσ|но|обаче)\b", t)
    return [p.strip() for p in parts if p and p.strip()]


def _rx(*alts):
    # patterns go through the same normalisation as the text (accents, Turkish i, final sigma...)
    return re.compile("|".join(normalize(a) for a in alts))


NEG = _rx(r"\bno\b", r"\bnot\b", r"\bnor\b", r"without", r"negative for", r"free of", r"absence",
          r"no evidence", r"\bsin\b", r"\bno se\b", r"ausencia", r"descart", r"\bkein", r"\bohne\b",
          r"\bnicht\b", r"ausgeschlossen", r"\bpas\b", r"\bsans\b", r"aucun", r"\bgeen\b", r"\bniet\b",
          r"zonder", r"\bbez\b", r"\bnema\b", r"\bnije\b", r"\bnisu\b", r"χωρισ", r"\bδεν\b", r"απουσια",
          r"ουδεμια", r"\bбез\b", r"\bняма\b", r"\bне\b", r"липса", r"\byok", r"izlenmedi", r"saptanmadi",
          r"gorulmedi", r"izlenmemis", r"saptanmamis", r"rastlanmadi", r"bulunmadi", r"degil",
          r"izlenmemekte", r"dikkati cekmedi")
NORMAL = _rx(r"normal", r"intact", r"unremarkable", r"preserved", r"integr", r"conservad",
             r"sin alteracion", r"sin lesion", r"habitual", r"intakt", r"unauffallig", r"regelrecht",
             r"erhalten", r"gaaf", r"dogal", r"olagan", r"saglam", r"uredn", r"ocuvan", r"uobicajen",
             r"φυσιολογ", r"ακεραι", r"нормал", r"запазен", r"интакт", r"б\.о", r"без патолог")
HEDGE = _rx(r"possib", r"probab", r"likely", r"suspic", r"suspect", r"suggest", r"cannot be (?:excluded|ruled out)",
            r"may represent", r"questionable", r"equivocal", r"posibl", r"sospech", r"sugest", r"no se descarta",
            r"moglich", r"wahrscheinl", r"verdacht", r"v\.a\.", r"fraglich", r"nicht auszuschliessen",
            r"evoqu", r"mogelijk", r"waarschijnlijk", r"olasi", r"muhtemel", r"supheli", r"dusundur",
            r"ekarte edilemez", r"moguc", r"vjerojatn", r"suspekt", r"sumnj", r"πιθαν", r"υποπτ",
            r"ενδεχομεν", r"възможн", r"вероятн", r"съмнение")
TEAR = _rx(r"tear", r"torn", r"ruptur", r"rotura", r"\broto\b", r"desgarr", r"\briss", r"einriss", r"abriss",
           r"dechir", r"fissur", r"scheur", r"yirti", r"kopu", r"devamsiz", r"kesinti", r"puknu", r"razder",
           r"prekid", r"ρηξ", r"ρηγμ", r"руптур", r"разкъс", r"скъс", r"discontinu", r"disrupt", r"deficien",
           r"not visuali", r"nonvisuali", r"no se visualiza", r"bucket", r"\basa de\b", r"korbhenkel",
           r"anse de seau", r"flap", r"colgajo", r"lappen", r"lambeau", r"flep", r"κρημν", r"фрагмент",
           r"grade (?:3|iii)\b", r"grado (?:3|iii)\b", r"grad (?:3|iii)\b", r"evre 3", r"stupnj?a? (?:3|iii)\b",
           r"βαθμου? (?:3|iii)\b", r"степен (?:3|iii)\b", r"lesion", r"lasion", r"lezyon", r"lezij",
           r"βλαβ", r"увреда", r"macerat", r"truncat", r"amput")
PARTIAL = _rx(r"partial", r"parcial", r"teil", r"partiel", r"partiele", r"gedeeltelijk", r"kismi",
              r"parcijal", r"djelomic", r"μερικ", r"частичн", r"incomplet", r"low.grade", r"bajo grado",
              r"sprain", r"esguince", r"distension", r"zerrung", r"elongation", r"entorse", r"distorsie",
              r"verstuik", r"burk", r"gerilme", r"istegnu", r"distenzij", r"διατασ", r"навяхван", r"разтяг")
HIGH = _rx(r"complete", r"full.thickness", r"complet", r"komplett", r"\btotal", r"volledig", r"tam kat",
           r"\btam\b", r"potpun", r"πληρ", r"пълн", r"high.grade", r"alto grado", r"hochgradig", r"haut grade",
           r"hooggradig", r"yuksek derece", r"visok\w* stup", r"υψηλ", r"висока степен")
MILD = _rx(r"mild", r"minimal", r"slight", r"\bearly", r"superficial", r"small", r"trace", r"physiolog",
           r"little", r"grade (?:1|2|i|ii)\b", r"grado (?:1|2|i|ii)\b", r"\bleve", r"incipiente", r"pequen",
           r"escas", r"minim", r"leicht", r"gering", r"beginnend", r"oberflachlich", r"diskret", r"klein",
           r"wenig", r"discret", r"leger", r"petit", r"\bpeu\b", r"faible", r"lichte", r"beetje", r"\bwat\b",
           r"hafif", r"\baz\b", r"erken", r"ufak", r"fizyolojik", r"blag", r"\bmal[ia]?\b", r"neznat",
           r"pocetn", r"ηπι", r"μικρ", r"ελαχιστ", r"αρχομεν", r"лек", r"малък", r"минимал", r"начален",
           r"незначит")
LARGE = _rx(r"moderate", r"large", r"massive", r"significant", r"marked", r"abundant", r"moderad",
            r"importante", r"significativ", r"\bgran", r"mittel", r"massig", r"deutlich", r"ausgepragt",
            r"\bgross", r"reichlich", r"modere", r"abondant", r"important", r"grand", r"matig", r"\bfors",
            r"uitgesprok", r"\bveel\b", r"groot", r"\borta\b", r"belirgin", r"fazla", r"\bcok\b", r"umjeren",
            r"obilan", r"izrazen", r"velik", r"znacajn", r"μετρι", r"αφθον", r"σημαντικ", r"μεγαλ",
            r"умерен", r"значителен", r"голям", r"изразен", r"обилен")

ANAT = {
    "ACL": _rx(r"anterior cruciate", r"\bacl\b", r"cruzado anterior", r"\blcae?\b", r"vordere\w* kreuzband",
               r"\bvkb\b", r"croise anterieur", r"voorste kruisband", r"on capraz", r"\bocb\b",
               r"prednj\w* (?:ukrizen|kriz)", r"προσθι\w* χιαστ", r"предн\w* кръстн", r"crociato anteriore"),
    "MCL": _rx(r"medial collateral", r"\bmcl\b", r"colateral (?:medial|interno)", r"lateral interno",
               r"\blli\b", r"\blcm\b", r"innenband", r"mediale\w* (?:kollateral|seiten)band",
               r"collateral\w* medial", r"lateral interne", r"mediale collateral", r"binnenband",
               r"medial kollateral", r"\bmkl\b", r"ic yan bag", r"medijaln\w* kolateral", r"kolateraln\w* \w* medijal",
               r"εσω πλαγι", r"медиалн\w* колатерал", r"collaterale mediale"),
    "Medial Meniscus": _rx(r"medial menisc", r"menisc\w* (?:medial|interno|interne)", r"menisco interno",
                           r"innenmenisk", r"mediale\w* menisk", r"menisque (?:medial|interne)",
                           r"mediale meniscus", r"binnenmeniscus", r"medial menisk", r"ic menisk",
                           r"medijaln\w* menisk", r"menisk\w* medijal", r"εσω μηνισκ", r"μηνισκ\w* εσω",
                           r"медиалн\w* менискус", r"менискус\w* медиал", r"menisco mediale"),
    "Lateral Meniscus": _rx(r"lateral menisc", r"menisc\w* (?:lateral|externo|externe)", r"menisco externo",
                            r"aussenmenisk", r"laterale\w* menisk", r"menisque (?:lateral|externe)",
                            r"laterale meniscus", r"buitenmeniscus", r"lateral menisk", r"dis menisk",
                            r"lateraln\w* menisk", r"menisk\w* lateral", r"εξω μηνισκ", r"μηνισκ\w* εξω",
                            r"латералн\w* менискус", r"менискус\w* латерал", r"menisco laterale"),
}
# group mentions that can only make a finding NEGATIVE ("cruciate ligaments intact", "menisci normal")
GROUP = {
    "ACL": _rx(r"cruciate ligaments", r"ligamentos cruzados", r"kreuzbander", r"ligaments croises",
               r"kruisbanden", r"capraz baglar", r"krizni ligamenti", r"ukrizen\w* sveze", r"χιαστοι",
               r"кръстни връзки", r"\bcruciates\b"),
    "MCL": _rx(r"collateral ligaments", r"ligamentos colaterales", r"seitenbander", r"kollateralbander",
               r"ligaments collateraux", r"collaterale banden", r"kollateral bag", r"kolateralni ligamenti",
               r"πλαγιοι συνδεσμοι", r"колатерални"),
    "Medial Meniscus": _rx(r"\bmenisci\b", r"meniscos", r"menisken", r"menisques", r"menisci", r"meniskusler",
                           r"menisküsler", r"menisci", r"μηνισκοι", r"менискуси", r"\bmenisk[ia]\b"),
}
GROUP["Lateral Meniscus"] = GROUP["Medial Meniscus"]
CART = _rx(r"chondr", r"condr", r"kondr", r"χονδρ", r"хондр", r"хрущял", r"hrskavic", r"kikirdak", r"knorpel",
           r"kraakbeen", r"cartilag", r"osteoarthr", r"arthros", r"artros", r"gonarthr", r"gonartr",
           r"οστεοαρθρ", r"αρθροσ", r"остеоартр", r"артроз", r"osteophyt", r"osteofit", r"οστεοφυτ", r"остеофит",
           r"denud", r"bone.on.bone", r"eburn", r"joint space narrow", r"pinzamiento", r"gelenkspaltversch",
           r"pincement", r"gewrichtsspleet")
OA_SEVERE = _rx(r"osteoarthr", r"arthros", r"artros", r"gonarthr", r"gonartr", r"οστεοαρθρ", r"αρθροσ",
                r"остеоартр", r"артроз", r"full.thickness", r"grade (?:3|4|iii|iv)\b", r"grado (?:3|4|iii|iv)\b",
                r"grad (?:3|4|iii|iv)\b", r"evre (?:3|4)", r"stupnj?a? (?:3|4|iii|iv)\b", r"βαθμου? (?:3|4|iii|iv)\b",
                r"степен (?:3|4|iii|iv)\b", r"denud", r"bone.on.bone", r"exposed bone", r"eburn",
                r"cartilage loss", r"loss of cartilage", r"perdida (?:de|del) cartilago", r"knorpelverlust",
                r"knorpelglatze", r"perte cartilag", r"kraakbeenverlies", r"kikirdak kaybi", r"gubitak hrskavic",
                r"απωλεια χονδρ", r"загуба на хрущял", r"изтънен", r"severe", r"advanced", r"sever[ao]",
                r"avanzad", r"schwer", r"fortgeschritten", r"hochgradig", r"avance", r"ernstig", r"gevorderd",
                r"ileri", r"siddetli", r"tesk", r"uznapredoval", r"σοβαρ", r"προχωρημεν", r"тежк", r"напреднал",
                r"moderate", r"modere", r"moderad", r"mittelgradig", r"matig", r"orta derece", r"umjeren",
                r"μετρι", r"умерен")
COMPART = {
    "Medial OA": _rx(r"\bmedial", r"\binterno", r"\binterne", r"\binnen", r"\bmediale", r"medijal", r"\bεσω",
                     r"медиал"),
    "Lateral OA": _rx(r"\blateral", r"\bexterno", r"\bexterne", r"\baussen", r"\blaterale", r"lateraln",
                      r"\bεξω", r"латерал"),
    "PF OA": _rx(r"patel", r"rotul", r"troch", r"trocle", r"femoropatel", r"retropatel", r"kneecap",
                 r"επιγονατ", r"τροχιλ", r"пател", r"трохле", r"femoro.?patell"),
}
TRICOMP = _rx(r"tricompart", r"three.compartment", r"drei.?kompartiment", r"uc kompartman", r"trikompart",
              r"τριδιαμερισ", r"трикомпарт", r"pancompart", r"all (?:three )?compartments")
STRIP_FOR_OA = _rx(r"\w*menisc\w*", r"\w*menisk\w*", r"\w*μηνισκ\w*", r"\w*менискус\w*",
                   r"collateral\w*", r"colateral\w*", r"kollateral\w*", r"kolateral\w*", r"retinac\w*")
EFFUSION = _rx(r"effusion", r"derrame", r"erguss", r"epanchement", r"hydrops", r"gewrichtsvocht", r"vochtcollectie",
               r"intra.?articula\w* (?:fluid|vocht|liquido)", r"efuzyon", r"efuzy", r"eklem\w* (?:ici )?(?:sivi|mayi)",
               r"\bmayi", r"sivi artis", r"izljev", r"συλλογ", r"излив", r"versamento")
SYNOVITIS = _rx(r"synovit", r"sinovit", r"synovial (?:thicken|prolif|hypertroph)", r"engrosamiento sinovial",
                r"synovia\w*verdick", r"synoviale verdick", r"epaississement synovial", r"sinovyal kalinl",
                r"sinovijaln\w* zadeb", r"συνοβιτ", r"υμενιτ", r"синовит", r"hoffa\w* (?:synovit|inflam)")
BAKER = _rx(r"baker", r"popliteal cyst", r"quiste popl", r"kyste popl", r"poplitea\w* (?:cyst|zyste|kist|cist)",
            r"knieholtecyst", r"ιγνυακ\w* κυστ", r"подколенн\w* кист", r"бейкер", r"μπεικερ",
            r"gastrocnem\w*.?semimembran", r"semimembran\w*.?gastrocnem")
CONTUSION_STRONG = _rx(r"contusi", r"bone bruise", r"\bbruis", r"kontusion", r"kneuzing", r"contusie", r"kontuzyon",
                       r"kontuzij", r"θλασ", r"контузи", r"microfractur", r"mikrofraktur", r"trabecular fractur",
                       r"trabekul\w* frakt")
MARROW_EDEMA = _rx(r"marrow (?:o)?edema", r"bone (?:o)?edema", r"edema oseo", r"edema de (?:la )?medula",
                   r"edema medular", r"knochenmark\w*(?:o|oe)dem", r"knochen(?:o|oe)dem", r"(?:o|oe)deme osseu",
                   r"(?:o|oe)deme medullaire", r"beenmerg\w*(?:o|oe)deem", r"bot(?:o|oe)deem", r"kemik ilig\w* odem",
                   r"kemik iligi", r"edem\w* kost", r"kostan\w* edem", r"edem kostane", r"οιδημα\w* (?:του )?(?:οστικου )?μυελ",
                   r"μυελικ\w* οιδημα", r"костномозъч\w* едем", r"костен оток", r"edema subcondral")
DEGEN_CTX = _rx(r"degenerat", r"subchondral", r"subcondral", r"arthros", r"artros", r"osteoarthr", r"reactiv",
                r"reaktiv", r"geod", r"insuffic", r"stress", r"overload", r"sobrecarga", r"uberlastung",
                r"surcharge", r"overbelast", r"dejenerat", r"εκφυλισ", r"υποχονδρ", r"дегенератив", r"субхондрал")
FRACTURE = _rx(r"fractur", r"fraktur", r"fractuur", r"\bkirik", r"\bfraktur", r"prijelom", r"καταγμ", r"фрактур",
               r"счупв", r"avulsi", r"segond", r"impaction")
OLD = _rx(r"\bold\b", r"healed", r"chronic", r"remote", r"antigu", r"consolidad", r"\balt", r"verheilt", r"ancienne",
          r"consolid", r"\boud", r"genezen", r"\beski", r"iyiles", r"zarasl", r"παλαι", r"стар", r"зараснал",
          r"\bprior\b", r"previous", r"post.?op", r"hardware", r"screw", r"tornillo", r"schraube", r"\bvis\b",
          r"\bvida", r"vijak", r"osteosynth", r"insuffic", r"stress")


def _status_from(clause_hits: list[str]) -> str:
    order = ["positive", "probable", "equivocal", "below_threshold", "negative"]
    hits = [h for h in clause_hits if h]
    return min(hits, key=order.index) if hits else "not_mentioned"


def _graded(c: str, sev_large=LARGE, sev_small=MILD) -> str:
    if NEG.search(c):
        return "negative"
    if HEDGE.search(c):
        return "equivocal"
    if sev_large.search(c):
        return "positive"
    if sev_small.search(c):
        return "below_threshold"
    return "probable"


def regex_statuses(report: str) -> dict:
    cl = clauses(normalize(report))
    hits = {lab: [] for lab in LABELS}
    for c in cl:
        neg = bool(NEG.search(c))
        # ligaments + menisci: anatomy word and tear word in the same clause
        for lab in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"):
            if ANAT[lab].search(c):
                if TEAR.search(c) and not neg:
                    if lab in ("ACL", "MCL") and PARTIAL.search(c) and not HIGH.search(c):
                        hits[lab].append("equivocal" if lab == "ACL" else "below_threshold")
                    elif lab.endswith("Meniscus") and MILD.search(c) and not HIGH.search(c):
                        hits[lab].append("equivocal")
                    else:
                        hits[lab].append("probable" if HEDGE.search(c) else "positive")
                elif neg or NORMAL.search(c):
                    hits[lab].append("negative")
                else:
                    hits[lab].append("below_threshold" if lab.endswith("Meniscus") else "negative")
            elif lab in GROUP and GROUP[lab].search(c) and (neg or NORMAL.search(c)) and not TEAR.search(c):
                hits[lab].append("negative")
        # osteoarthritis per compartment
        if CART.search(c):
            oc = STRIP_FOR_OA.sub(" ", c)
            tri = bool(TRICOMP.search(oc))
            for lab in ("Medial OA", "Lateral OA", "PF OA"):
                if tri or COMPART[lab].search(oc):
                    if neg and not OA_SEVERE.search(oc):
                        hits[lab].append("negative")
                    elif OA_SEVERE.search(oc) and not MILD.search(oc):
                        hits[lab].append("probable" if HEDGE.search(oc) else "positive")
                    elif OA_SEVERE.search(oc):
                        hits[lab].append("equivocal")
                    else:
                        hits[lab].append("below_threshold")
        if EFFUSION.search(c):
            hits["Effusion"].append(_graded(c))
        if SYNOVITIS.search(c):
            hits["Synovitis"].append("negative" if neg else ("equivocal" if HEDGE.search(c) else "positive"))
        if BAKER.search(c):
            hits["Baker's"].append(_graded(c))
        if CONTUSION_STRONG.search(c):
            hits["Contusion"].append("negative" if neg else ("probable" if HEDGE.search(c) else "positive"))
        elif MARROW_EDEMA.search(c):
            hits["Contusion"].append("negative" if neg else
                                     ("below_threshold" if DEGEN_CTX.search(c) else "probable"))
        if FRACTURE.search(c):
            if neg:
                hits["Fracture"].append("negative")
            elif OLD.search(c):
                hits["Fracture"].append("below_threshold")
            else:
                hits["Fracture"].append("probable" if HEDGE.search(c) else "positive")
    return {lab: _status_from(h) for lab, h in hits.items()}
