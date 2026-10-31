import sys
import io
import json
import random
import re
import threading
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import fire
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

OUT = settings.DATA_DIR / "other"
UA = "twicetold-research"
SEED = 0
YEARS = range(2018, 2027)
MAX_PAPERS = 500
MIN_CHARS = 100
WORKERS = 4
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = UA
        retry = Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        _local.s = s
    return _local.s


class Blocked(Exception):
    pass


def fetch(url, path, params=None):
    path = Path(path)
    if path.exists():
        return path.read_bytes() or None
    r = session().get(url, params=params, timeout=120)
    if r.status_code in (404, 410):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return None
    if r.status_code == 403 or "cf-mitigated" in r.headers:
        raise Blocked(url)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.rename(path)
    return r.content


def pmap(fn, items, desc, workers=WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc, mininterval=10))


def clean(text):
    text = (text or "").encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return re.sub(r"\s+", " ", text).strip()


def shuffled(items):
    items = sorted(items)
    random.Random(SEED).shuffle(items)
    return items


def finalize(source, reviews, papers, out, max_papers=MAX_PAPERS, stats=None):
    rv = pd.DataFrame([dict(r, text=clean(r["text"])) for r in reviews])
    pp = pd.DataFrame(papers).drop_duplicates("paper_id")
    rv["n_chars"] = rv.text.str.len()
    stats = dict(stats or {}, parsed_reviews=len(rv), short_dropped=int((rv.n_chars < MIN_CHARS).sum()))
    rv = rv[rv.n_chars >= MIN_CHARS]
    n0 = len(rv)
    rv = rv.drop_duplicates(["paper_id", "text"])
    stats["dup_dropped"] = n0 - len(rv)
    print(source, stats)
    rv["review_time"] = pd.to_datetime(rv.review_time, utc=True, format="ISO8601").dt.tz_localize(None)
    first = rv.groupby("paper_id").review_time.min().dt.year
    rv["year"] = rv.paper_id.map(first).fillna(rv.year)
    rv = rv[rv.year.notna()].astype({"year": int})
    n = rv.groupby("paper_id").size()
    keep = n[n >= 2].index
    rv = rv[rv.paper_id.isin(keep) & rv.year.isin(YEARS)]
    yr = rv.groupby("paper_id").year.first()
    chosen = []
    for y, ids in yr.groupby(yr):
        ids = shuffled(ids.index.tolist())
        chosen += ids[: max_papers(y) if callable(max_papers) else max_papers]
    rv = rv[rv.paper_id.isin(chosen)].reset_index(drop=True)
    pp = pp[pp.paper_id.isin(chosen)].copy()
    pp["year"] = pp.paper_id.map(yr)
    pp["n_reviews"] = pp.paper_id.map(rv.groupby("paper_id").size())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rv.to_parquet(out / f"{source}_reviews.parquet")
    pp.to_parquet(out / f"{source}_papers.parquet")
    (out / f"{source}_stats.json").write_text(json.dumps(stats))
    print(rv.groupby("year").agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")))


def epmc_search(query, cache):
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    hits, cursor = [], "*"
    with tqdm(desc=f"europepmc {cache.parent.name} {cache.stem}", unit="page") as bar:
        while True:
            r = session().get(f"{EPMC}/search", params=dict(query=query, format="json", resultType="lite", pageSize=1000, cursorMark=cursor), timeout=120)
            r.raise_for_status()
            d = r.json()
            hits += [dict(pmcid=h.get("pmcid"), doi=h.get("doi"), year=h.get("pubYear"), journal=h.get("journalTitle"), title=h.get("title"))
                     for h in d["resultList"]["result"]]
            bar.update()
            if d.get("nextCursorMark") in (None, cursor) or not d["resultList"]["result"]:
                break
            cursor = d["nextCursorMark"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(hits))
    return hits


def epmc_xml(pmcid, folder):
    try:
        raw = fetch(f"{EPMC}/{pmcid}/fullTextXML", Path(folder) / f"{pmcid}.xml")
    except requests.RequestException as e:
        print("skip", pmcid, type(e).__name__)
        (Path(folder) / f"{pmcid}.xml").write_bytes(b"")
        return None
    if not raw:
        return None
    try:
        return ET.fromstring(raw)
    except ET.ParseError:
        return None


def text_of(el):
    return clean("".join(el.itertext())) if el is not None else ""


def licence_of(root):
    for el in root.iter():
        for v in [el.text or ""] + list(el.attrib.values()):
            m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/?([\d.]+)?", v)
            if m:
                return f"CC {m.group(2).upper()} {m.group(3) or ''}".strip().replace("CC ZERO", "CC0")
    lic = text_of(root.find(".//license"))
    return "CC BY (stated in text)" if "Creative Commons Attribution" in lic else (lic[:80] or None)


def collect(candidates, parse, desc, batch=100):
    papers, reviews, found, i = [], [], 0, 0
    cands = shuffled(candidates)
    for i in range(0, len(cands), batch):
        for res in pmap(parse, cands[i:i + batch], desc):
            if res is None:
                continue
            p, rs = res
            rs = [r for r in rs if len(clean(r["text"])) >= MIN_CHARS]
            papers.append(p)
            reviews += rs
            found += len(rs) >= 2
        if found >= MAX_PAPERS:
            break
    print(desc, "scanned", min(len(cands), i + batch), "of", len(cands), "qualifying", found)
    return papers, reviews


def epmc_collect(source, journals, parse, extra=""):
    query = " OR ".join(f'JOURNAL:"{j}"' for j in journals)
    papers, reviews = [], []
    for y in YEARS:
        hits = epmc_search(f"({query}) AND PUB_YEAR:{y}{extra}", settings.RAW_JOURNAL_REVIEWS_DIR / source / f"search_{y}.json")
        p, r = collect([h["pmcid"] for h in hits if h["pmcid"]], parse, f"{source} {y}")
        papers += p
        reviews += r
    return papers, reviews


SCIPOST_API = "https://scipost.org/api/submissions/"
SCIPOST_PAGE = 100
SCIPOST_SCALE = {"top": 6, "high": 5, "good": 4, "ok": 3, "low": 2, "poor": 1}
SCIPOST_CRITERIA = ["validity", "significance", "originality", "clarity", "formatting", "grammar"]
SCIPOST_FIELDS = [("strengths", "STRENGTHS"), ("weaknesses", "WEAKNESSES"), ("report", "REPORT"), ("requested_changes", "REQUESTED CHANGES")]


def scipost_page(offset):
    raw = fetch(f"{SCIPOST_API}?format=json&limit={SCIPOST_PAGE}&offset={offset}", settings.RAW_JOURNAL_REVIEWS_DIR / "scipost" / f"page_{offset:06d}.json")
    return json.loads(raw)["results"]


def scipost(out=OUT):
    count = json.loads(fetch(f"{SCIPOST_API}?format=json&limit=1", settings.RAW_JOURNAL_REVIEWS_DIR / "scipost" / "count.json"))["count"]
    subs = [s for p in pmap(scipost_page, range(0, count, SCIPOST_PAGE), "scipost pages") for s in p]
    reviews, papers = [], []
    for s in subs:
        if s["is_resubmission_of"]:
            continue
        pid = s["identifier"]
        for r in s["reports"]:
            if r["status"] != "vetted" or r["report_type"] != "report_normal":
                continue
            text = "\n\n".join(f"[{tag}] {clean(r[k])}" for k, tag in SCIPOST_FIELDS if clean(r.get(k)))
            scores = [SCIPOST_SCALE[r[c]] for c in SCIPOST_CRITERIA[:4] if r.get(c) in SCIPOST_SCALE]
            reviews.append(dict(conf="SciPost", field=(s.get("acad_field") or "physics").lower(), journal=s["submitted_to"],
                                year=int(s["submission_date"][:4]), paper_id=pid, review_id=r["doi_string"] or f"{pid}#r{r['report_nr']}",
                                reviewer=f"Reviewer {r['report_nr']}", rating=round(sum(scores) / len(scores)) if scores else None,
                                rating_raw=";".join(f"{c}={r[c]}" for c in SCIPOST_CRITERIA if r.get(c) not in (None, "", "-")) or None,
                                text=text, review_time=r["date_submitted"]))
        papers.append(dict(conf="SciPost", field=(s.get("acad_field") or "physics").lower(), journal=s["submitted_to"],
                           paper_id=pid, title=s["title"], licence="CC BY 4.0"))
    finalize("scipost", reviews, papers, out)


F1000_JOURNALS = ["F1000Research", "Wellcome Open Research", "Open Research Europe"]
F1000_ANSWER = re.compile(r"^(yes|no|partly|not applicable|i cannot comment.*|no source data required|n/?a)\.?$", re.I)
F1000_TAIL = re.compile(r"^(i|we) (confirm that|have read this submission)", re.I)


def f1000_rating(label):
    label = label.lower()
    if "reservation" in label:
        return 2
    if "not approved" in label or "reject" in label:
        return 1
    if "approve" in label:
        return 3
    return None


def f1000_report_text(body):
    for parent in list(body.iter()):
        for ch in list(parent):
            if ch.tag == "ref-list":
                parent.remove(ch)
    paras = [text_of(el) for el in body]
    paras = paras[:-1] if paras and paras[-1] == "References" else paras
    keep, i = [], 0
    while i < len(paras):
        p = paras[i]
        nxt = paras[i + 1] if i + 1 < len(paras) else ""
        if p.endswith("?") and F1000_ANSWER.match(nxt):
            i += 2
        elif p.lower().startswith("reviewer expertise"):
            i += 1 if p.rstrip(":").lower() != "reviewer expertise" else 2
        elif F1000_TAIL.match(p):
            i += 1
        else:
            keep.append(p)
            i += 1
    return "\n\n".join(keep)


def f1000_parse(pmcid):
    root = epmc_xml(pmcid, settings.RAW_JOURNAL_REVIEWS_DIR / "f1000")
    if root is None:
        return None
    front = root.find("front/article-meta")
    doi = front.findtext("article-id[@pub-id-type='doi']") or pmcid
    pid = re.sub(r"\.\d+$", "", doi)
    journal = root.findtext("front/journal-meta/journal-title-group/journal-title")
    paper = dict(conf="F1000", field="biomed", journal=journal, paper_id=pid,
                 title=text_of(front.find("title-group/article-title")), licence=licence_of(root.find("front")))
    reviews = []
    for sa in root.iter("sub-article"):
        if sa.get("article-type") not in ("reviewer-report", "peer-review", "ref-report"):
            continue
        meta = sa.find("front-stub")
        meta = meta if meta is not None else sa.find("front/article-meta")
        if meta is None or not re.search(r"version 1\b", text_of(meta.find("title-group/article-title")), re.I):
            continue
        label = next((cm.findtext("meta-value") for cm in sa.iter("custom-meta") if cm.findtext("meta-name") == "recommendation"), None)
        if label is None:
            m = re.search(r"Recommendation:\s*([^.]+)", text_of(sa.find(".//notes")) or "")
            label = m.group(1) if m else None
        if label is None and sa.find("body") is not None:
            m = re.search(r"I confirm that I have read this submission[^.]*", text_of(sa.find("body")))
            s = m.group(0) if m else ""
            label = ("Not Approved" if "do not consider" in s else "Approved with Reservations" if "reservations" in s
                     else "Approved" if "acceptable scientific standard" in s else None)
            label = f"{label} (from confirmation statement)" if label else None
        d = meta.find("pub-date")
        when = f"{d.findtext('year')}-{int(d.findtext('month') or 1):02d}-{int(d.findtext('day') or 1):02d}" if d is not None else None
        body = sa.find("body")
        reviews.append(dict(conf="F1000", field="biomed", journal=journal, year=int(when[:4]) if when else None, paper_id=pid,
                            review_id=meta.findtext("article-id[@pub-id-type='doi']") or f"{pmcid}#{sa.get('id')}",
                            reviewer=None, rating=f1000_rating(label) if label else None, rating_raw=label,
                            text=f1000_report_text(body) if body is not None else "", review_time=when))
    return paper, reviews


def f1000(out=OUT):
    papers, reviews = epmc_collect("f1000", F1000_JOURNALS, f1000_parse)
    uniq = {}
    for r in sorted(reviews, key=lambda r: (r["paper_id"], r["review_time"] or "", r["review_id"])):
        uniq.setdefault(r["review_id"], r)
    k = Counter()
    for r in uniq.values():
        k[r["paper_id"]] += 1
        r["reviewer"] = f"Reviewer {k[r['paper_id']]}"
    finalize("f1000", list(uniq.values()), papers, out)


BMC_URL = "https://huggingface.co/datasets/JerMa88/BMC_Medicine_Peer_Reviews/resolve/main/train.parquet"


def fuzzy(phrase):
    return r"\s?".join(re.escape(c) if c not in " '" else (r"\s+" if c == " " else r"[’']") for c in phrase)


BMC_START = re.compile(fuzzy("Reviewer's report") + r"\s?:", re.I)
BMC_STOPS = re.compile("|".join(fuzzy(s) for s in [
    "Are the methods appropriate", "Are you able to assess", "Quality of written English", "Declaration of competing",
    "Level of interest", "Statistical review", "Statement on potential review bias", "I agree to the open peer review",
    "Reviewer Publication Consent"]))
BMC_DATE = re.compile(r"Date\s?:\s*(\d{1,2}\s+\w{3}\s+\d{4})")
BMC_TITLE = re.compile(r"Title\s?:\s*(.*?)\s*Version\s?:", re.S)


def bmc_body(t):
    m = BMC_START.search(t)
    if not m:
        return None
    t = t[m.end():]
    s = BMC_STOPS.search(t)
    return t[: s.start()] if s else t


def bmc(out=OUT):
    fetch(BMC_URL, settings.RAW_JOURNAL_REVIEWS_DIR / "bmc" / "train.parquet")
    df = pd.read_parquet(settings.RAW_JOURNAL_REVIEWS_DIR / "bmc" / "train.parquet", columns=["title", "review", "review_src", "year"])
    df = df[df.review_src.str.contains(r"_V0_R\d+\.pdf$")]
    reviews, papers = [], []
    for r in df.itertuples():
        doi = r.title.removeprefix("Review for ").strip()
        n = re.search(r"_R(\d+)\.pdf$", r.review_src).group(1)
        d = BMC_DATE.search(r.review)
        title = BMC_TITLE.search(r.review)
        reviews.append(dict(conf="BMC-Med", field="medicine", journal="BMC Medicine", year=r.year, paper_id=doi,
                            review_id=r.review_src.removesuffix(".pdf"), reviewer=f"Reviewer {n}", rating=None, rating_raw=None,
                            text=bmc_body(r.review) or "", review_time=pd.to_datetime(d.group(1), format="%d %b %Y", errors="coerce") if d else None))
        papers.append(dict(conf="BMC-Med", field="medicine", journal="BMC Medicine", paper_id=doi,
                           title=re.sub(r"\s+", " ", title.group(1)) if title else None, licence="CC BY 4.0"))
    print("V0 reports", len(reviews), "with empty body (no header or text only in an attachment)", sum(not x["text"] for x in reviews))
    finalize("bmc", reviews, papers, out)


PLOS_JOURNALS = ["PLoS One", "PLoS Biol", "PLoS Comput Biol", "PLoS Genet", "PLoS Med", "PLoS Pathog", "PLoS Negl Trop Dis",
                 "PLOS Glob Public Health", "PLOS Digit Health"]
PLOS_SEP = "**********"
PLOS_START = re.compile(r"Reviewers?['’]? (?:comments|Responses to Questions)|Comments to the Author", re.I)
PLOS_END = re.compile("|".join([r"PLOS authors have the option to publish", r"Do you want your identity to be public",
                                r"\[NOTE: If reviewer comments", r"While revising your submission", r"\[Note: HTML markup is below",
                                r"To ensure your figures meet", r"Figure Files:", r"Data Requirements:",
                                r"Reproducibility:(?=\W*To enhance)", r"Any attachments provided with reviews", r"(?m:^-{10,}[- ]*$)",
                                r"General editorial requests"]), re.I)
PLOS_REV = re.compile(r"Reviewer\s*#\s*(\d+)\s*:\s*")
PLOS_ANS = re.compile(r"(Yes|No|Partly|N/A|I Don['’]t Know)[.\s]*", re.I)
PLOS_SCORE = {"yes": 3, "partly": 2, "no": 1}
PLOS_SKIPPED = Counter()
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def plos_letters(root):
    out = []
    for sa in root.iter("sub-article"):
        m = re.fullmatch(r"Decision Letter (\d+)", text_of(sa.find(".//article-title")))
        if m:
            out.append((int(m.group(1)), sa))
    return sorted(out, key=lambda t: t[0])


def plos_split_letter(text):
    m = PLOS_START.search(text)
    text = text[m.start():] if m else text
    texts, answers = {}, {}
    for k, block in enumerate(text.split(PLOS_SEP), 1):
        parts = PLOS_REV.split(block)
        header, entries = parts[0], list(zip(parts[1::2], parts[2::2]))
        if not entries or re.search(r"identity to be public|option to publish", header, re.I):
            continue
        if all(len(t.strip()) <= 20 for _, t in entries):
            q = re.findall(r"(?:^|\n)(\d+)\.\s", header)
            for n, t in entries:
                a = PLOS_ANS.fullmatch(t.strip())
                if a:
                    answers.setdefault(int(n), {})[int(q[-1]) if q else k] = (a.group(1), "technically sound" in header.lower())
            continue
        for n, t in entries:
            end = PLOS_END.search(t)
            t = (t[: end.start()] if end else t).strip()
            texts[int(n)] = (texts.get(int(n), "") + "\n\n" + t).strip()
    return texts, answers


def plos_parse(pmcid):
    root = epmc_xml(pmcid, settings.RAW_JOURNAL_REVIEWS_DIR / "plos")
    if root is None:
        return None
    if any(text_of(sa.find(".//article-title")) == "Author response to previous submission" for sa in root.iter("sub-article")):
        PLOS_SKIPPED["resubmission"] += 1
        return None
    sa = next((sa for _, sa in plos_letters(root) if PLOS_REV.search(text_of(sa.find("body")))), None)
    if sa is None:
        return None
    meta = root.find("front/article-meta")
    doi = meta.findtext("article-id[@pub-id-type='doi']") or pmcid
    journal = root.findtext("front/journal-meta/journal-title-group/journal-title")
    field = "multidisciplinary" if journal and journal.upper() == "PLOS ONE" else "biomed"
    body = sa.find("body")
    paras = [re.sub(r"</?ctrl>", "", text_of(el)) for el in body]
    date = next((text_of(el) for el in sa.iter("named-content") if el.get("content-type") == "letter-date"), None)
    if not date:
        m = re.match(r"\s*(\d{1,2} \w{3} \d{4})", paras[0] if paras else "")
        rec = meta.find("history/date[@date-type='received']")
        date = m.group(1) if m else (f"{rec.findtext('day')} {MONTHS[int(rec.findtext('month')) - 1]} {rec.findtext('year')}" if rec is not None else None)
    texts, answers = plos_split_letter("\n\n".join(paras))
    paper = dict(conf="PLOS", field=field, journal=journal, paper_id=doi, title=text_of(meta.find("title-group/article-title")),
                 licence=licence_of(root.find("front")), n_split=len(texts))
    reviews = []
    for n, t in sorted(texts.items()):
        ans = answers.get(n, {})
        sound = next((a for a, is_sound in ans.values() if is_sound), None)
        reviews.append(dict(conf="PLOS", field=field, journal=journal, year=None, paper_id=doi, review_id=f"{doi}.r001#R{n}",
                            reviewer=f"Reviewer {n}", rating=PLOS_SCORE.get(sound.lower()) if sound else None,
                            rating_raw=";".join(f"Q{q}={a}" for q, (a, _) in sorted(ans.items())) or None,
                            text=t, review_time=date))
    return paper, reviews


def plos(out=OUT):
    papers, reviews = epmc_collect("plos", PLOS_JOURNALS, plos_parse, extra=' AND "Decision Letter 0"')
    for r in reviews:
        r["review_time"] = pd.to_datetime(r["review_time"], format="%d %b %Y", errors="coerce")
    split2 = sum(p["n_split"] >= 2 for p in papers)
    print("skipped papers:", dict(PLOS_SKIPPED), "letters split into 2+ reviewers:", split2, "of", len(papers))
    finalize("plos", reviews, papers, out, stats=dict(PLOS_SKIPPED, letters=len(papers), letters_split_2plus=split2))


COP = settings.RAW_JOURNAL_REVIEWS_DIR / "copernicus"
COP_CROSSREF = "https://api.crossref.org/prefixes/10.5194/works"
COP_SCORE = {"accept": 4, "minor-revision": 3, "major-revision": 2, "reject": 1}
COP_SKIPPED = Counter()
COP_PDF_JUNK = re.compile(r"^(C\d+|Interactive|comment|Interactive comment|Printer-friendly version|Discussion paper|Full screen / Esc)$")


def copernicus_cap(year):
    return 500 if year >= 2022 else 300


def crossref_index():
    done = COP / "crossref" / "done.json"
    if not done.exists():
        cursor, n = "*", 0
        with tqdm(desc="crossref pages", unit="page") as bar:
            while True:
                r = session().get(COP_CROSSREF, params={"filter": "type:peer-review,from-created-date:2018-01-01", "rows": 1000, "cursor": cursor},
                                  timeout=120)
                r.raise_for_status()
                msg = r.json()["message"]
                rows = [dict(doi=i["DOI"].lower(), created=i["created"]["date-time"], url=i.get("resource", {}).get("primary", {}).get("URL"),
                             rec=i.get("review", {}).get("recommendation"), kind=i.get("review", {}).get("type"))
                        for i in msg["items"]]
                (COP / "crossref").mkdir(parents=True, exist_ok=True)
                (COP / "crossref" / f"page_{n:04d}.json").write_text(json.dumps(rows))
                n += 1
                bar.update()
                if not msg["items"] or not msg.get("next-cursor"):
                    break
                cursor = msg["next-cursor"]
        done.write_text(json.dumps(n))
    rows = [r for p in sorted((COP / "crossref").glob("page_*.json")) for r in json.loads(p.read_text())]
    return [r for r in rows if re.search(r"-rc\d+$", r["doi"])]


def copernicus_page(url, ms):
    html = fetch(url, COP / "html" / f"{ms}.html")
    if html and b"http-equiv=refresh" in html[:2000]:
        target = re.search(rb'url=([^"\'>]+)', html).group(1).decode()
        html = fetch(target, COP / "html" / f"{ms}.disc.html")
    return html.decode("utf-8", "replace") if html else None


def copernicus_pdf_text(url, name):
    try:
        raw = fetch(url, COP / "pdf" / name)
    except (Blocked, requests.ConnectionError):
        COP_SKIPPED["pdf fetch error"] += 1
        return ""
    if not raw:
        return ""
    try:
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    except Exception as e:
        print("pdf error", name, e)
        return ""
    m = re.search(r"Received and published:.*\n", text)
    text = text[m.end():] if m else text
    end = text.rfind("Interactive comment on ")
    text = text[:end] if end > 0 else text
    text = "\n".join(line for line in text.splitlines() if not COP_PDF_JUNK.match(line.strip()))
    return unicodedata.normalize("NFKC", re.sub(r"-\n(\w)", r"\1", text))


def copernicus_parse(item):
    ms, recs = item
    m = re.match(rf"https?://[^/]+/preprints/(?:\d{{4}}/)?{re.escape(ms)}(?=[/#]|$)", recs[0]["url"] or "")
    url = (m.group(0) if m else f"https://{ms.split('-')[0]}.copernicus.org/preprints/{ms}") + "/"
    try:
        html = copernicus_page(url, ms)
    except Blocked:
        COP_SKIPPED["blocked"] += 1
        return None
    except requests.ConnectionError:
        COP_SKIPPED["connection error"] += 1
        return None
    if not html:
        COP_SKIPPED["no page"] += 1
        return None
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    journal, _, title = title.partition(" - Peer review - ") if " - Peer review - " in title else title.partition(" - ")
    by_rc = {r["doi"].rsplit("-", 1)[1].upper(): r for r in recs}
    reviews, seen, names = [], set(), {}
    for tag in soup.select("span.part-title strong"):
        rc = tag.get_text(strip=True)
        if not re.fullmatch(r"RC\d+", rc):
            continue
        content = tag.find_parent("span").find_next_sibling("span")
        head = clean(content.get_text(" ", strip=True)) if content else ""
        m = re.search(r",\s*([^,]+?),\s*(\d{1,2} \w{3} \d{4})", head)
        who, when = (m.group(1), m.group(2)) if m else ("?", None)
        n = re.search(r"Referee #(\d+)", who)
        label = f"Referee {n.group(1)}" if n else names.setdefault(who, f"Referee {len(names) + 101}")
        if label in seen:
            continue
        li = tag.find_parent("li")
        body = li.find("div", class_=re.compile(rf"\b{rc}\b")) if li else None
        text = body.get_text("\n", strip=True) if body else ""
        text = re.split(r"\n\s*Citation\s*:", text)[0]
        pdfs = [urljoin(url, a["href"].strip()) for a in (li.find_all("a", href=True) if li else [])
                if re.search(rf"-{rc}(-supplement)?\.pdf$", a["href"].strip())]
        if len(clean(text)) < 300 and pdfs:
            text = (text + "\n\n" + "\n\n".join(copernicus_pdf_text(p, p.rsplit("/", 1)[1]) for p in pdfs)).strip()
        seen.add(label)
        rec = by_rc.get(rc, {})
        reviews.append(dict(conf="Copernicus", field="geoscience", journal=journal.strip(), year=int(rec.get("created", "0000")[:4]),
                            paper_id=f"10.5194/{ms}", review_id=f"10.5194/{ms}-{rc.lower()}", reviewer=label,
                            rating=COP_SCORE.get(rec.get("rec")), rating_raw=rec.get("rec"), text=text,
                            review_time=str(datetime.strptime(when, "%d %b %Y").date()) if when else rec.get("created")))
    paper = dict(conf="Copernicus", field="geoscience", journal=journal.strip(), paper_id=f"10.5194/{ms}", title=title.strip(),
                 licence="CC BY 4.0")
    return paper, reviews


def copernicus(out=OUT):
    by_ms = defaultdict(list)
    for r in crossref_index():
        by_ms[r["doi"].removeprefix("10.5194/").rsplit("-", 1)[0]].append(r)
    by_year = defaultdict(list)
    for ms, recs in by_ms.items():
        if len(recs) >= 2:
            by_year[int(min(r["created"] for r in recs)[:4])].append(ms)
    papers, reviews = [], []
    for y in YEARS:
        cands = shuffled(by_year.get(y, []))[: int(copernicus_cap(y) * 1.2)]
        print("copernicus", y, "candidates with 2+ RC", len(by_year.get(y, [])), "fetching", len(cands))
        items = [(ms, sorted(by_ms[ms], key=lambda r: r["created"])) for ms in cands]
        for res in pmap(copernicus_parse, items, f"copernicus {y}", workers=2):
            if res:
                papers.append(res[0])
                reviews += res[1]
    print("skipped papers:", dict(COP_SKIPPED))
    finalize("copernicus", reviews, papers, out, max_papers=copernicus_cap, stats=dict(COP_SKIPPED))


ICML_FIELDS = {2025: [("summary", "SUMMARY"), ("claims_and_evidence", "CLAIMS AND EVIDENCE"),
                      ("methods_and_evaluation_criteria", "METHODS AND EVALUATION"), ("theoretical_claims", "THEORY"),
                      ("experimental_designs_or_analyses", "EXPERIMENTS"), ("supplementary_material", "SUPPLEMENT"),
                      ("relation_to_broader_scientific_literature", "RELATED WORK"),
                      ("essential_references_not_discussed", "MISSING REFERENCES"),
                      ("other_strengths_and_weaknesses", "STRENGTHS AND WEAKNESSES"),
                      ("other_comments_or_suggestions", "OTHER"), ("questions_for_authors", "QUESTIONS")],
               2026: [("summary", "SUMMARY"), ("strengths_and_weaknesses", "STRENGTHS AND WEAKNESSES"),
                      ("key_questions_for_authors", "QUESTIONS"), ("limitations", "LIMITATIONS")]}


def val(content, key):
    v = content.get(key)
    return v.get("value") if isinstance(v, dict) else v


def icml_download(year, raw):
    import openreview

    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net",
                                             username=settings.OPENREVIEW_USERNAME, password=settings.OPENREVIEW_PASSWORD)
    notes = client.get_all_notes(content={"venueid": f"ICML.cc/{year}/Conference"}, details="replies")
    with open(raw, "w") as f:
        f.writelines(json.dumps(dict(id=n.id, content=n.content, replies=n.details.get("replies", []))) + "\n" for n in notes)
    print("submissions", len(notes))


def icml_reviews(year, raw):
    rows = []
    with open(raw) as f:
        for line in tqdm(f, desc=f"icml {year} submissions"):
            s = json.loads(line)
            for r in s["replies"]:
                if not any(inv.endswith("/Official_Review") for inv in r.get("invitations", [])):
                    continue
                c = r["content"]
                text = "\n\n".join(f"[{tag}]\n{val(c, k)}" for k, tag in ICML_FIELDS[year] if val(c, k))
                rows.append(dict(conf="ICML", year=year, paper_id=s["id"], review_id=r["id"], text=text,
                                 rating=val(c, "overall_recommendation"), confidence=val(c, "confidence"),
                                 primary_area=val(s["content"], "primary_area"), review_time=r.get("cdate"), modified_time=r.get("mdate")))
    return pd.DataFrame(rows)


def icml(year=2026, out=settings.DATA_DIR):
    raw = settings.DATA_DIR / f"icml{year}_raw.jsonl"
    if not raw.exists():
        icml_download(year, raw)
    df = icml_reviews(year, raw)
    df.to_parquet(Path(out) / f"icml{year}_reviews.parquet")
    print(len(df), "reviews of", df.paper_id.nunique(), "papers")


if __name__ == "__main__":
    fire.Fire(dict(scipost=scipost, f1000=f1000, bmc=bmc, plos=plos, copernicus=copernicus, icml=icml))
