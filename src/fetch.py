import io
import json
import random
import re
import sys
import threading
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import fire
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.exceptions import ResponseError
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

UA = "twicetold-research"
SEED = 0
YEARS = range(2018, 2027)
MAX_PAPERS = 500
MIN_CHARS = 100
MIN_REVIEWS = 2
WORKERS = 4
TIMEOUT = 120
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EPMC_BROKEN_RECORD = ResponseError.SPECIFIC_ERROR.format(status_code=500)
OPENREVIEW_API = "https://api2.openreview.net"

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


def fetch(url, path):
    path = Path(path)
    if path.exists():
        return path.read_bytes() or None
    r = session().get(url, timeout=TIMEOUT)
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
    before_dedup = len(rv)
    rv = rv.drop_duplicates(["paper_id", "text"])
    stats["dup_dropped"] = before_dedup - len(rv)
    print(source, stats)
    rv["review_time"] = pd.to_datetime(rv.review_time, utc=True, format="ISO8601").dt.tz_localize(None)
    first_review_year = rv.groupby("paper_id").review_time.min().dt.year
    rv["year"] = rv.paper_id.map(first_review_year).fillna(rv.year)
    rv = rv[rv.year.notna()].astype({"year": int})
    reviews_per_paper = rv.groupby("paper_id").size()
    reviewed_papers = reviews_per_paper[reviews_per_paper >= MIN_REVIEWS].index
    rv = rv[rv.paper_id.isin(reviewed_papers) & rv.year.isin(YEARS)]
    paper_year = rv.groupby("paper_id").year.first()
    chosen = []
    for year, ids in paper_year.groupby(paper_year):
        year_cap = max_papers(year) if callable(max_papers) else max_papers
        chosen += shuffled(ids.index.tolist())[:year_cap]
    rv = rv[rv.paper_id.isin(chosen)].reset_index(drop=True)
    pp = pp[pp.paper_id.isin(chosen)].copy()
    pp["year"] = pp.paper_id.map(paper_year)
    pp["n_reviews"] = pp.paper_id.map(rv.groupby("paper_id").size())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rv.to_parquet(out / f"{source}_reviews.parquet")
    pp.to_parquet(out / f"{source}_papers.parquet")
    (out / f"{source}_stats.json").write_text(json.dumps(stats))
    print(rv.groupby("year").agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")))


def epmc_hit(hit):
    return dict(
        pmcid=hit.get("pmcid"),
        doi=hit.get("doi"),
        year=hit.get("pubYear"),
        journal=hit.get("journalTitle"),
        title=hit.get("title"),
    )


def epmc_search(query, cache):
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    hits, cursor = [], "*"
    with tqdm(desc=f"europepmc {cache.parent.name} {cache.stem}", unit="page") as bar:
        while True:
            r = session().get(
                f"{EPMC}/search",
                params=dict(query=query, format="json", resultType="lite", pageSize=1000, cursorMark=cursor),
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            page = r.json()
            results = page["resultList"]["result"]
            hits += [epmc_hit(h) for h in results]
            bar.update()
            if page.get("nextCursorMark") in (None, cursor) or not results:
                break
            cursor = page["nextCursorMark"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(hits))
    return hits


def epmc_xml(pmcid, folder):
    path = Path(folder) / f"{pmcid}.xml"
    try:
        raw = fetch(f"{EPMC}/{pmcid}/fullTextXML", path)
    except requests.RequestException as e:
        if not isinstance(e, requests.HTTPError) and EPMC_BROKEN_RECORD not in str(e):
            raise
        print("skip", pmcid, type(e).__name__)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
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
    candidates = shuffled(candidates)
    papers, reviews = [], []
    scanned = 0
    found = 0
    for start in range(0, len(candidates), batch):
        chunk = candidates[start : start + batch]
        scanned += len(chunk)
        for res in pmap(parse, chunk, desc):
            if res is None:
                continue
            paper, paper_reviews = res
            paper_reviews = [r for r in paper_reviews if len(clean(r["text"])) >= MIN_CHARS]
            papers.append(paper)
            reviews += paper_reviews
            found += len(paper_reviews) >= MIN_REVIEWS
        if found >= MAX_PAPERS:
            break
    print(desc, "scanned", scanned, "of", len(candidates), "qualifying", found)
    return papers, reviews


def epmc_collect(source, journals, parse, extra=""):
    query = " OR ".join(f'JOURNAL:"{j}"' for j in journals)
    papers, reviews = [], []
    for y in YEARS:
        hits = epmc_search(f"({query}) AND PUB_YEAR:{y}{extra}", settings.RAW_JOURNAL_REVIEWS_DIR / source / f"search_{y}.json")
        year_papers, year_reviews = collect([h["pmcid"] for h in hits if h["pmcid"]], parse, f"{source} {y}")
        papers += year_papers
        reviews += year_reviews
    return papers, reviews


SCIPOST_RAW_DIR = settings.RAW_JOURNAL_REVIEWS_DIR / "scipost"
SCIPOST_API = "https://scipost.org/api/submissions/"
SCIPOST_PAGE = 100
SCIPOST_SCALE = {"top": 6, "high": 5, "good": 4, "ok": 3, "low": 2, "poor": 1}
SCIPOST_CRITERIA = ["validity", "significance", "originality", "clarity", "formatting", "grammar"]
SCIPOST_RATED = SCIPOST_CRITERIA[:4]
SCIPOST_FIELDS = [
    ("strengths", "STRENGTHS"),
    ("weaknesses", "WEAKNESSES"),
    ("report", "REPORT"),
    ("requested_changes", "REQUESTED CHANGES"),
]


def scipost_page(offset):
    raw = fetch(f"{SCIPOST_API}?format=json&limit={SCIPOST_PAGE}&offset={offset}", SCIPOST_RAW_DIR / f"page_{offset:06d}.json")
    return json.loads(raw)["results"]


def scipost(out=settings.OTHER_REVIEWS_DIR):
    count = json.loads(fetch(f"{SCIPOST_API}?format=json&limit=1", SCIPOST_RAW_DIR / "count.json"))["count"]
    pages = pmap(scipost_page, range(0, count, SCIPOST_PAGE), "scipost pages")
    submissions = list({sub["identifier"]: sub for page in pages for sub in page}.values())
    missing = max(count - len(submissions), 0)
    if missing:
        print(f"scipost: offset pages miss {missing} of {count} submissions; the API order is not stable")
    reviews, papers = [], []
    for sub in submissions:
        if sub["is_resubmission_of"]:
            continue
        pid = sub["identifier"]
        venue = dict(conf="SciPost", field=(sub.get("acad_field") or "physics").lower(), journal=sub["submitted_to"])
        for report in sub["reports"]:
            if report["status"] != "vetted" or report["report_type"] != "report_normal":
                continue
            fields = {tag: clean(report.get(key)) for key, tag in SCIPOST_FIELDS}
            fields = {tag: value for tag, value in fields.items() if value}
            is_stub = len(" ".join(fields.values())) < MIN_CHARS
            text = "" if is_stub else "\n\n".join(f"[{tag}] {value}" for tag, value in fields.items())
            scores = [SCIPOST_SCALE[report[c]] for c in SCIPOST_RATED if report.get(c) in SCIPOST_SCALE]
            rating = round(sum(scores) / len(scores)) if scores else None
            rating_raw = ";".join(f"{c}={report[c]}" for c in SCIPOST_CRITERIA if report.get(c) not in (None, "", "-")) or None
            reviews.append(
                dict(
                    venue,
                    year=int(sub["submission_date"][:4]),
                    paper_id=pid,
                    review_id=report["doi_string"] or f"{pid}#r{report['report_nr']}",
                    reviewer=f"Reviewer {report['report_nr']}",
                    rating=rating,
                    rating_raw=rating_raw,
                    text=text,
                    review_time=report["date_submitted"],
                )
            )
        papers.append(dict(venue, paper_id=pid, title=sub["title"], licence="CC BY 4.0"))
    finalize("scipost", reviews, papers, out, stats=dict(api_missing=missing))


F1000_JOURNALS = ["F1000Research", "Wellcome Open Research", "Open Research Europe"]
F1000_ANSWER = re.compile(r"^(yes|no|partly|not applicable|i cannot comment.*|no source data required|n/?a)\.?$", re.I)
F1000_TAIL = re.compile(r"^(i|we) (confirm that|have read this submission)", re.I)
F1000_CONFIRMATION_LABELS = [
    ("do not consider", "Not Approved"),
    ("reservations", "Approved with Reservations"),
    ("acceptable scientific standard", "Approved"),
]


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
        for child in list(parent):
            if child.tag == "ref-list":
                parent.remove(child)
    paras = [text_of(el) for el in body]
    if paras and paras[-1] == "References":
        paras = paras[:-1]
    keep = []
    i = 0
    while i < len(paras):
        para = paras[i]
        next_para = paras[i + 1] if i + 1 < len(paras) else ""
        if para.endswith("?") and F1000_ANSWER.match(next_para):
            i += 2
        elif para.lower().startswith("reviewer expertise"):
            i += 2 if para.rstrip(":").lower() == "reviewer expertise" else 1
        elif F1000_TAIL.match(para):
            i += 1
        else:
            keep.append(para)
            i += 1
    return "\n\n".join(keep)


def f1000_label(sa, body):
    recommendations = (cm.findtext("meta-value") for cm in sa.iter("custom-meta") if cm.findtext("meta-name") == "recommendation")
    label = next(recommendations, None)
    if label is not None:
        return label
    notes = re.search(r"Recommendation:\s*([^.]+)", text_of(sa.find(".//notes")) or "")
    if notes:
        return notes.group(1)
    if body is None:
        return None
    statement = re.search(r"I confirm that I have read this submission[^.]*", text_of(body))
    confirmation = statement.group(0) if statement else ""
    stated = next((name for phrase, name in F1000_CONFIRMATION_LABELS if phrase in confirmation), None)
    return f"{stated} (from confirmation statement)" if stated else None


def f1000_date(meta):
    date = meta.find("pub-date")
    if date is None:
        return None
    month = int(date.findtext("month") or 1)
    day = int(date.findtext("day") or 1)
    return f"{date.findtext('year')}-{month:02d}-{day:02d}"


def f1000_parse(pmcid):
    root = epmc_xml(pmcid, settings.RAW_JOURNAL_REVIEWS_DIR / "f1000")
    if root is None:
        return None
    front = root.find("front/article-meta")
    doi = front.findtext("article-id[@pub-id-type='doi']") or pmcid
    pid = re.sub(r"\.\d+$", "", doi)
    journal = root.findtext("front/journal-meta/journal-title-group/journal-title")
    venue = dict(conf="F1000", field="biomed", journal=journal)
    paper = dict(
        venue,
        paper_id=pid,
        title=text_of(front.find("title-group/article-title")),
        licence=licence_of(root.find("front")),
    )
    reviews = []
    for sa in root.iter("sub-article"):
        if sa.get("article-type") not in ("reviewer-report", "peer-review", "ref-report"):
            continue
        meta = sa.find("front-stub")
        if meta is None:
            meta = sa.find("front/article-meta")
        if meta is None or not re.search(r"version 1\b", text_of(meta.find("title-group/article-title")), re.I):
            continue
        body = sa.find("body")
        label = f1000_label(sa, body)
        when = f1000_date(meta)
        reviews.append(
            dict(
                venue,
                year=int(when[:4]) if when else None,
                paper_id=pid,
                review_id=meta.findtext("article-id[@pub-id-type='doi']") or f"{pmcid}#{sa.get('id')}",
                reviewer=None,
                rating=f1000_rating(label) if label else None,
                rating_raw=label,
                text=f1000_report_text(body) if body is not None else "",
                review_time=when,
            )
        )
    return paper, reviews


def f1000(out=settings.OTHER_REVIEWS_DIR):
    papers, reviews = epmc_collect("f1000", F1000_JOURNALS, f1000_parse)
    by_review_id = {}
    for r in sorted(reviews, key=lambda r: (r["paper_id"], r["review_time"] or "", r["review_id"])):
        by_review_id.setdefault(r["review_id"], r)
    reviewers_seen = Counter()
    for r in by_review_id.values():
        reviewers_seen[r["paper_id"]] += 1
        r["reviewer"] = f"Reviewer {reviewers_seen[r['paper_id']]}"
    finalize("f1000", list(by_review_id.values()), papers, out)


BMC_URL = "https://huggingface.co/datasets/JerMa88/BMC_Medicine_Peer_Reviews/resolve/main/train.parquet"
BMC_VENUE = dict(conf="BMC-Med", field="medicine", journal="BMC Medicine")
BMC_REVIEWER = re.compile(r"_R(\d+)\.pdf$")
BMC_STOP_PHRASES = [
    "Are the methods appropriate",
    "Are you able to assess",
    "Quality of written English",
    "Declaration of competing",
    "Level of interest",
    "Statistical review",
    "Statement on potential review bias",
    "I agree to the open peer review",
    "Reviewer Publication Consent",
]


def fuzzy(phrase):
    special = {" ": r"\s+", "'": r"[’']"}
    return r"\s?".join(special.get(c, re.escape(c)) for c in phrase)


BMC_START = re.compile(fuzzy("Reviewer's report") + r"\s?:", re.I)
BMC_STOPS = re.compile("|".join(fuzzy(phrase) for phrase in BMC_STOP_PHRASES))
BMC_DATE = re.compile(r"Date\s?:\s*(\d{1,2}\s+\w{3}\s+\d{4})")
BMC_TITLE = re.compile(r"Title\s?:\s*(.*?)\s*Version\s?:", re.S)


def bmc_body(text):
    start = BMC_START.search(text)
    if not start:
        return None
    text = text[start.end() :]
    stop = BMC_STOPS.search(text)
    return text[: stop.start()] if stop else text


def bmc(out=settings.OTHER_REVIEWS_DIR):
    raw = settings.RAW_JOURNAL_REVIEWS_DIR / "bmc" / "train.parquet"
    fetch(BMC_URL, raw)
    df = pd.read_parquet(raw, columns=["title", "review", "review_src", "year"])
    df = df[df.review_src.str.contains(r"_V0_R\d+\.pdf$")]
    reviews, papers = [], []
    for row in df.itertuples():
        doi = row.title.removeprefix("Review for ").strip()
        reviewer_no = BMC_REVIEWER.search(row.review_src).group(1)
        date = BMC_DATE.search(row.review)
        when = pd.to_datetime(date.group(1), format="%d %b %Y", errors="coerce") if date else None
        title = BMC_TITLE.search(row.review)
        reviews.append(
            dict(
                BMC_VENUE,
                year=row.year,
                paper_id=doi,
                review_id=row.review_src.removesuffix(".pdf"),
                reviewer=f"Reviewer {reviewer_no}",
                rating=None,
                rating_raw=None,
                text=bmc_body(row.review) or "",
                review_time=when,
            )
        )
        papers.append(
            dict(
                BMC_VENUE,
                paper_id=doi,
                title=re.sub(r"\s+", " ", title.group(1)) if title else None,
                licence="CC BY 4.0",
            )
        )
    print(
        "V0 reports",
        len(reviews),
        "with empty body (no header or text only in an attachment)",
        sum(not x["text"] for x in reviews),
    )
    finalize("bmc", reviews, papers, out)


PLOS_JOURNALS = [
    "PLoS One",
    "PLoS Biol",
    "PLoS Comput Biol",
    "PLoS Genet",
    "PLoS Med",
    "PLoS Pathog",
    "PLoS Negl Trop Dis",
    "PLOS Glob Public Health",
    "PLOS Digit Health",
]
PLOS_SEP = "**********"
PLOS_START = re.compile(r"Reviewers?['’]? (?:comments|Responses to Questions)|Comments to the Author", re.I)
PLOS_END = re.compile(
    "|".join(
        [
            r"PLOS authors have the option to publish",
            r"Do you want your identity to be public",
            r"\[NOTE: If reviewer comments",
            r"While revising your submission",
            r"\[Note: HTML markup is below",
            r"To ensure your figures meet",
            r"Figure Files:",
            r"Data Requirements:",
            r"Reproducibility:(?=\W*To enhance)",
            r"Any attachments provided with reviews",
            r"(?m:^-{10,}[- ]*$)",
            r"General editorial requests",
        ]
    ),
    re.I,
)
PLOS_REV = re.compile(r"Reviewer\s*#\s*(\d+)\s*:\s*")
PLOS_ANS = re.compile(r"(Yes|No|Partly|N/A|I Don['’]t Know)[.\s]*", re.I)
PLOS_MAX_ANSWER_CHARS = 20
PLOS_QUESTION_NO = re.compile(r"(?:^|\n)(\d+)\.\s")
PLOS_CONSENT_QUESTION = re.compile(r"identity to be public|option to publish", re.I)
PLOS_SCORE = {"yes": 3, "partly": 2, "no": 1}
PLOS_EARLIER_REVIEWS = re.compile(r"Responses? to Reviewer Comments|Reviewer\s*#?\s*\d", re.I)
PLOS_RESPONSE_FILE = re.compile(r"Response to Reviewers\.\w+", re.I)
PLOS_SKIPPED = Counter()
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def plos_letters(root):
    letters = []
    for sa in root.iter("sub-article"):
        m = re.fullmatch(r"Decision Letter (\d+)", text_of(sa.find(".//article-title")))
        if m:
            letters.append((int(m.group(1)), sa))
    return sorted(letters, key=lambda t: t[0])


def plos_responses_between(root, first_letter, letter):
    responses, started = [], False
    for sa in root.iter("sub-article"):
        if sa is letter:
            break
        started = started or sa is first_letter
        if started and text_of(sa.find(".//article-title")).startswith("Author response"):
            responses.append(PLOS_RESPONSE_FILE.sub("", text_of(sa.find("body"))))
    return " ".join(responses)


def plos_split_letter(text):
    start = PLOS_START.search(text)
    text = text[start.start() :] if start else text
    texts, answers = {}, {}
    for block_no, block in enumerate(text.split(PLOS_SEP), 1):
        parts = PLOS_REV.split(block)
        header = parts[0]
        entries = [(int(reviewer), entry) for reviewer, entry in zip(parts[1::2], parts[2::2])]
        if not entries or PLOS_CONSENT_QUESTION.search(header):
            continue
        if all(len(entry.strip()) <= PLOS_MAX_ANSWER_CHARS for _, entry in entries):
            numbers = PLOS_QUESTION_NO.findall(header)
            question = int(numbers[-1]) if numbers else block_no
            is_soundness_question = "technically sound" in header.lower()
            for reviewer, entry in entries:
                answer = PLOS_ANS.fullmatch(entry.strip())
                if answer:
                    answers.setdefault(reviewer, {})[question] = (answer.group(1), is_soundness_question)
            continue
        for reviewer, entry in entries:
            end = PLOS_END.search(entry)
            entry = (entry[: end.start()] if end else entry).strip()
            texts[reviewer] = (texts.get(reviewer, "") + "\n\n" + entry).strip()
    return texts, answers


def plos_parse(pmcid):
    root = epmc_xml(pmcid, settings.RAW_JOURNAL_REVIEWS_DIR / "plos")
    if root is None:
        return None
    if any(text_of(sa.find(".//article-title")) == "Author response to previous submission" for sa in root.iter("sub-article")):
        PLOS_SKIPPED["resubmission"] += 1
        return None
    letters = [sa for _, sa in plos_letters(root)]
    letter = next((sa for sa in letters if PLOS_REV.search(text_of(sa.find("body")))), None)
    if letter is None:
        return None
    if PLOS_EARLIER_REVIEWS.search(plos_responses_between(root, letters[0], letter)):
        PLOS_SKIPPED["earlier_reviews"] += 1
        return None
    meta = root.find("front/article-meta")
    doi = meta.findtext("article-id[@pub-id-type='doi']") or pmcid
    journal = root.findtext("front/journal-meta/journal-title-group/journal-title")
    field = "multidisciplinary" if journal and journal.upper() == "PLOS ONE" else "biomed"
    venue = dict(conf="PLOS", field=field, journal=journal)
    body = letter.find("body")
    paras = [re.sub(r"</?ctrl>", "", text_of(el)) for el in body]
    date = next((text_of(el) for el in letter.iter("named-content") if el.get("content-type") == "letter-date"), None)
    if not date:
        opening = re.match(r"\s*(\d{1,2} \w{3} \d{4})", paras[0] if paras else "")
        received = meta.find("history/date[@date-type='received']")
        if opening:
            date = opening.group(1)
        elif received is not None:
            date = f"{received.findtext('day')} {MONTHS[int(received.findtext('month')) - 1]} {received.findtext('year')}"
        else:
            date = None
    texts, answers = plos_split_letter("\n\n".join(paras))
    paper = dict(
        venue,
        paper_id=doi,
        title=text_of(meta.find("title-group/article-title")),
        licence=licence_of(root.find("front")),
        n_split=len(texts),
    )
    reviews = []
    for reviewer, text in sorted(texts.items()):
        reviewer_answers = answers.get(reviewer, {})
        soundness = next((a for a, is_soundness_question in reviewer_answers.values() if is_soundness_question), None)
        reviews.append(
            dict(
                venue,
                year=None,
                paper_id=doi,
                review_id=f"{doi}.r001#R{reviewer}",
                reviewer=f"Reviewer {reviewer}",
                rating=PLOS_SCORE.get(soundness.lower()) if soundness else None,
                rating_raw=";".join(f"Q{q}={a}" for q, (a, _) in sorted(reviewer_answers.items())) or None,
                text=text,
                review_time=date,
            )
        )
    return paper, reviews


def plos(out=settings.OTHER_REVIEWS_DIR):
    papers, reviews = epmc_collect("plos", PLOS_JOURNALS, plos_parse, extra=' AND "Decision Letter 0"')
    for r in reviews:
        r["review_time"] = pd.to_datetime(r["review_time"], format="%d %b %Y", errors="coerce")
    split2 = sum(p["n_split"] >= 2 for p in papers)
    print("skipped papers:", dict(PLOS_SKIPPED), "letters split into 2+ reviewers:", split2, "of", len(papers))
    finalize("plos", reviews, papers, out, stats=dict(PLOS_SKIPPED, letters=len(papers), letters_split_2plus=split2))


COP_RAW_DIR = settings.RAW_JOURNAL_REVIEWS_DIR / "copernicus"
COP_CROSSREF = "https://api.crossref.org/prefixes/10.5194/works"
COP_SCORE = {"accept": 4, "minor-revision": 3, "major-revision": 2, "reject": 1}
COP_SKIPPED = Counter()
COP_PDF_JUNK = re.compile(
    r"^(C\d+|Interactive|comment|Interactive comment|Printer-friendly version|Discussion paper|Full screen / Esc)$"
)
COP_OVERSAMPLE = 1.2
COP_MIN_HTML_CHARS = 300
COP_NAMED_REFEREE_FIRST = 101
COP_MANUSCRIPT_FOOTER = "Discussion started:"
COP_MIN_MANUSCRIPT_PAGES = 3
COP_SUPPLEMENT_NOTE = re.compile(
    r"(?:The comment was uploaded in the form of a supplement|Please also note the supplement to this comment)[.:]?(?:\s*https?://\S+)?"
)
COP_PREPRINT_SERVER = "egusphere"
COP_JOURNAL_NAMES = {"esurf": "ESurf", "gchron": "GChron", "sand": "SaND"}


def copernicus_cap(year):
    return 500 if year >= 2022 else 300


def crossref_row(item):
    return dict(
        doi=item["DOI"].lower(),
        created=item["created"]["date-time"],
        url=item.get("resource", {}).get("primary", {}).get("URL"),
        rec=item.get("review", {}).get("recommendation"),
        kind=item.get("review", {}).get("type"),
    )


def crossref_index():
    folder = COP_RAW_DIR / "crossref"
    done = folder / "done.json"
    if not done.exists():
        cursor, pages = "*", 0
        with tqdm(desc="crossref pages", unit="page") as bar:
            while True:
                r = session().get(
                    COP_CROSSREF,
                    params={"filter": "type:peer-review,from-created-date:2018-01-01", "rows": 1000, "cursor": cursor},
                    timeout=TIMEOUT,
                )
                r.raise_for_status()
                msg = r.json()["message"]
                rows = [crossref_row(item) for item in msg["items"]]
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"page_{pages:04d}.json").write_text(json.dumps(rows))
                pages += 1
                bar.update()
                if not msg["items"] or not msg.get("next-cursor"):
                    break
                cursor = msg["next-cursor"]
        done.write_text(json.dumps(pages))
    rows = [r for page in sorted(folder.glob("page_*.json")) for r in json.loads(page.read_text())]
    return [r for r in rows if re.search(r"-rc\d+$", r["doi"])]


def copernicus_page(url, ms):
    html = fetch(url, COP_RAW_DIR / "html" / f"{ms}.html")
    if html and b"http-equiv=refresh" in html[:2000]:
        target = re.search(rb'url=([^"\'>]+)', html).group(1).decode()
        html = fetch(target, COP_RAW_DIR / "html" / f"{ms}.disc.html")
    return html.decode("utf-8", "replace") if html else None


def copernicus_pdf_text(url, name):
    try:
        raw = fetch(url, COP_RAW_DIR / "pdf" / name)
    except Blocked, requests.ConnectionError:
        COP_SKIPPED["pdf fetch error"] += 1
        return ""
    if not raw:
        return ""
    try:
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages]
    except Exception as e:
        print("pdf error", name, e)
        return ""
    review_pages = [page for page in pages if COP_MANUSCRIPT_FOOTER not in page]
    if len(pages) - len(review_pages) >= COP_MIN_MANUSCRIPT_PAGES:
        COP_SKIPPED["pdf with manuscript pages"] += 1
        pages = review_pages
    text = "\n".join(pages)
    m = re.search(r"Received and published:.*\n", text)
    text = text[m.end() :] if m else text
    end = text.rfind("Interactive comment on ")
    text = text[:end] if end > 0 else text
    text = "\n".join(line for line in text.splitlines() if not COP_PDF_JUNK.match(line.strip()))
    return unicodedata.normalize("NFKC", re.sub(r"-\n(\w)", r"\1", text))


def copernicus_journal(ms, soup):
    prefix = ms.split("-")[0]
    if prefix == COP_PREPRINT_SERVER:
        final_paper = soup.find("a", string="Final revised paper", href=True)
        if final_paper is None:
            return "EGUsphere"
        prefix = urlparse(final_paper["href"]).hostname.split(".")[0]
    return COP_JOURNAL_NAMES.get(prefix, prefix.upper())


def copernicus_parse(item):
    ms, recs = item
    crossref_url = re.match(rf"https?://[^/]+/preprints/(?:\d{{4}}/)?{re.escape(ms)}(?=[/#]|$)", recs[0]["url"] or "")
    url = (crossref_url.group(0) if crossref_url else f"https://{ms.split('-')[0]}.copernicus.org/preprints/{ms}") + "/"
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
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    sep = " - Peer review - " if " - Peer review - " in page_title else " - "
    title = page_title.partition(sep)[2]
    pid = f"10.5194/{ms}"
    venue = dict(conf="Copernicus", field="geoscience", journal=copernicus_journal(ms, soup))
    by_rc = {r["doi"].rsplit("-", 1)[1].upper(): r for r in recs}
    reviews = []
    seen = set()
    names = {}
    for tag in soup.select("span.part-title strong"):
        rc = tag.get_text(strip=True)
        if not re.fullmatch(r"RC\d+", rc):
            continue
        content = tag.find_parent("span").find_next_sibling("span")
        head = clean(content.get_text(" ", strip=True)) if content else ""
        byline = re.search(r",\s*([^,]+?),\s*(\d{1,2} \w{3} \d{4})", head)
        who, when = byline.groups() if byline else ("?", None)
        referee_no = re.search(r"Referee #(\d+)", who)
        if referee_no:
            label = f"Referee {referee_no.group(1)}"
        else:
            label = names.setdefault(who, f"Referee {len(names) + COP_NAMED_REFEREE_FIRST}")
        if label in seen:
            continue
        li = tag.find_parent("li")
        body = li.find("div", class_=re.compile(rf"\b{rc}\b")) if li else None
        text = body.get_text("\n", strip=True) if body else ""
        text = re.split(r"\n\s*Citation\s*:", text)[0]
        hrefs = [a["href"].strip() for a in li.find_all("a", href=True)] if li else []
        pdfs = [urljoin(url, href) for href in hrefs if re.search(rf"-{rc}(-supplement)?\.pdf$", href)]
        if len(clean(text)) < COP_MIN_HTML_CHARS and pdfs:
            pdf_texts = [copernicus_pdf_text(pdf, pdf.rsplit("/", 1)[1]) for pdf in pdfs]
            text = (text + "\n\n" + "\n\n".join(pdf_texts)).strip()
        if len(clean(COP_SUPPLEMENT_NOTE.sub("", text))) < MIN_CHARS:
            text = ""
        seen.add(label)
        rec = by_rc.get(rc, {})
        reviews.append(
            dict(
                venue,
                year=int(rec.get("created", "0000")[:4]),
                paper_id=pid,
                review_id=f"{pid}-{rc.lower()}",
                reviewer=label,
                rating=COP_SCORE.get(rec.get("rec")),
                rating_raw=rec.get("rec"),
                text=text,
                review_time=str(datetime.strptime(when, "%d %b %Y").date()) if when else rec.get("created"),
            )
        )
    return dict(venue, paper_id=pid, title=title.strip(), licence="CC BY 4.0"), reviews


def copernicus(out=settings.OTHER_REVIEWS_DIR):
    by_ms = defaultdict(list)
    for r in crossref_index():
        ms = r["doi"].removeprefix("10.5194/").rsplit("-", 1)[0]
        by_ms[ms].append(r)
    by_year = defaultdict(list)
    for ms, recs in by_ms.items():
        if len(recs) >= MIN_REVIEWS:
            by_year[int(min(r["created"] for r in recs)[:4])].append(ms)
    papers, reviews = [], []
    for y in YEARS:
        year_ms = by_year.get(y, [])
        candidates = shuffled(year_ms)[: int(copernicus_cap(y) * COP_OVERSAMPLE)]
        print("copernicus", y, "candidates with 2+ RC", len(year_ms), "fetching", len(candidates))
        items = [(ms, sorted(by_ms[ms], key=lambda r: r["created"])) for ms in candidates]
        for res in pmap(copernicus_parse, items, f"copernicus {y}", workers=2):
            if res:
                paper, paper_reviews = res
                papers.append(paper)
                reviews += paper_reviews
    print("skipped papers:", dict(COP_SKIPPED))
    finalize("copernicus", reviews, papers, out, max_papers=copernicus_cap, stats=dict(COP_SKIPPED))


ICML_FIELDS = {
    2025: [
        ("summary", "SUMMARY"),
        ("claims_and_evidence", "CLAIMS AND EVIDENCE"),
        ("methods_and_evaluation_criteria", "METHODS AND EVALUATION"),
        ("theoretical_claims", "THEORY"),
        ("experimental_designs_or_analyses", "EXPERIMENTS"),
        ("supplementary_material", "SUPPLEMENT"),
        ("relation_to_broader_scientific_literature", "RELATED WORK"),
        ("essential_references_not_discussed", "MISSING REFERENCES"),
        ("other_strengths_and_weaknesses", "STRENGTHS AND WEAKNESSES"),
        ("other_comments_or_suggestions", "OTHER"),
        ("questions_for_authors", "QUESTIONS"),
    ],
    2026: [
        ("summary", "SUMMARY"),
        ("strengths_and_weaknesses", "STRENGTHS AND WEAKNESSES"),
        ("key_questions_for_authors", "QUESTIONS"),
        ("limitations", "LIMITATIONS"),
    ],
}


def value_of(content, key):
    v = content.get(key)
    return v.get("value") if isinstance(v, dict) else v


def icml_download(year, raw):
    import openreview

    client = openreview.api.OpenReviewClient(
        baseurl=OPENREVIEW_API, username=settings.OPENREVIEW_USERNAME, password=settings.OPENREVIEW_PASSWORD
    )
    notes = client.get_all_notes(content={"venueid": f"ICML.cc/{year}/Conference"}, details="replies")
    tmp = raw.with_suffix(raw.suffix + ".part")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(dict(id=n.id, content=n.content, replies=n.details.get("replies", []))) + "\n" for n in notes)
    tmp.rename(raw)
    print("submissions", len(notes))


def icml_reviews(year, raw):
    rows = []
    with open(raw) as f:
        for line in tqdm(f, desc=f"icml {year} submissions"):
            sub = json.loads(line)
            for reply in sub["replies"]:
                if not any(inv.endswith("/Official_Review") for inv in reply.get("invitations", [])):
                    continue
                content = reply["content"]
                text = "\n\n".join(
                    f"[{tag}]\n{value_of(content, key)}" for key, tag in ICML_FIELDS[year] if value_of(content, key)
                )
                rows.append(
                    dict(
                        conf="ICML",
                        year=year,
                        paper_id=sub["id"],
                        review_id=reply["id"],
                        text=text,
                        rating=value_of(content, "overall_recommendation"),
                        confidence=value_of(content, "confidence"),
                        primary_area=value_of(sub["content"], "primary_area"),
                        review_time=reply.get("cdate"),
                        modified_time=reply.get("mdate"),
                    )
                )
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
