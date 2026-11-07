# twicetold

## Code rules

1. No comments, no docstrings, no file docstrings, in any file (Python, shell, jinja2, TOML, YAML). The only
   allowed `#` line is a shebang on line 1. CI runs `scripts/check_no_comments.py` and fails on the first one.
   Explain a decision in the commit message, not in the code.
2. Configuration lives in `src/settings.py` (pydantic-settings, upper-case field names): paths, default model,
   reasoning efforts, prices, OpenRouter routes, API keys from `.env`. No module defines a path, a model name or
   reads `os.environ`.
3. Prompts are jinja2 templates in `src/prompts/`, rendered by `src/prompts.py`. No prompt text in Python.
4. Run chains are `scripts/extract_all.sh`, `scripts/release.sh` and `scripts/post.sh`. A new pass or post step goes
   there, not in a scratch script.
5. Always UV: `uv run python src/<module>.py <command>`. Keep CPU use low (`nice -n 19`, `OMP_NUM_THREADS=2`).
6. Model outputs are named by model short name (`claims_<model>.jsonl`, `clusters_<model>.jsonl`,
   `reports/<name>_<model>.csv`). One model has one claims file and one clusters file. A new model or effort setting
   gets a new short name in `settings.OPENROUTER_ROUTES`.
7. All imports sit at the top of the file. No import inside a function or a class. No code between imports.
   `src/` is importable through the editable install (`dev-mode-dirs` in `pyproject.toml`); never touch
   `sys.path`. CI runs ruff with `E402` and `PLC0415` and fails on the first one.
8. The Hugging Face dataset `settings.HF_DATASET_REPO` holds the reviews, papers, claims, panels and MELD scores, plus
   the working files under `artifacts/`. Analysis code reads them through `src/hub.py`; only the producer steps
   (`fetch`, `corpus`, `panels`, `extract`, `meld`, `areas`, `release`) read or write the local copies in `data/`.


## Writing guide

Sources:
- [SPJ] Peyton Jones. *How to write a great research paper.* Microsoft Research slides (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf).
- [LS] Lipton, Steinhardt. *Troubling Trends in Machine Learning Scholarship.* arXiv:1807.03341, 2018.
- [JS] Schulman. *An Opinionated Guide to ML Research.* 2020 (http://joschu.net/blog/opinionated-guide-ml-research.html). Mostly on research process, not writing; cited sparingly.
- [NN] Nanda. *Highly Opinionated Advice on How to Write ML Papers.* Alignment Forum, 2025.
- [CL] Pechmann, after Widom. *Tips for Writing Technical Papers (Checklist).* Hochschule Emden/Leer.

### 1. Core claims first

- Write the 1–3 claims before any prose. Everything else serves them; cut a sentence if it serves none [NN, SPJ: "one ping"].
- Give each claim a confidence level: systematic ("across all settings we checked"), hedged ("suggestive"), or narrow ("in these three settings") [NN]. Do not turn a hedged claim into a systematic one anywhere.
- Give each claim 1–3 key experiments; put the rest in the appendix [NN].

### 2. Abstract

- Order: one sentence of context, one of the gap, then the claims with one concrete number each, then what the reader should do with the result [NN]. CL says use little or no background; NN allows two sentences to orient a cold reader. In a short paper, use one.
- State the standard of evidence in the last sentence: "provides evidence that", not "shows that", unless the result survives every check [NN, LS].
- Name the paper's genre at once, e.g. "a measurement study of N items from sources S" [NN].

### 3. Introduction

- Open with the problem, not the field. Use a concrete example rather than "X is important [1–6]" [SPJ: molehills, not mountains].
- CL's six questions, in this order: what the problem is; why it matters; why it is hard; what prior work missed; what we did and its limits; our contributions [CL].
- Write contributions as a bullet list of refutable claims, each with a forward reference: "A is X% higher than B (§4, Fig. 1)", not "we study A" [SPJ].
- Budget: about half a page in a short paper.

### 4. Calibrated language

- Separate what you measured from why you think it happened. A measured change is a finding; a mechanism you did not test is speculation. Label speculation as such and keep it in one paragraph [LS §3.1].
- Test: "Would I rely on this explanation to predict new data?" If not, it is speculation [LS §5.1].
- No suggestive terms. A proxy is not the thing it proxies: a detector's flag is not ground truth [LS §3.4.1]. Define each new term once, with one example.
- No overloaded terms: words such as "collapse" or "bias" carry other meanings. Use a neutral word and give its formula [LS §3.4.2–3].
- No mathiness: equations only where they make a definition exact; no theorem or notation you use only once [LS §3.3].

### 5. Experiments as evidence

- Each experiment answers "which hypothesis does this rule out?" If it rules out nothing, cut it [NN].
- Baselines are mandatory and must be strong [NN, CL].
- Every number has a sample size and a 95% CI. CIs come from the right unit: the cluster, when observations share one [NN; Figure guide §7].
- Never let a p-value between .01 and .05 be the only support for a central claim; report the effect size with its CI instead [NN].
- Show robustness to each analysis choice you made [LS §5.1, NN]. Give one sentence per check in the main text and the full table in the appendix.
- Mark which analyses you planned before seeing the data and which are post hoc [NN].
- Do not cherry-pick qualitative examples; show random ones, or say how you chose them [NN].
- A simple statistic that answers the question beats a complex model that impresses [JS: a small gain must come with small complexity; LS §3.2].

### 6. Main text vs appendix

- Main text: everything needed to understand and believe the core claims [CL]. Appendix: everything a typical reader skips [CL, NN].
- NN says to polish the appendix little; CL says it must still fully back the claims. Follow CL for numbers and tables, and NN for prose.
- A reader must be able to replicate the work from the main text plus the appendix: prompts, model versions, thresholds, data dates, code link [NN].
- Robustness checks, validation and extra settings go to the appendix, each with one forward reference from the main text.

### 7. Related work

- SPJ: put it late, after the idea, because a comparison is hard to read before the reader knows the idea. CL: put it early only if it is short or the paper must defend its position. NN: second to last, unless prior work motivates the paper.
- In a short paper, a few sentences in the introduction can place the work, with the full section at the end. Each paragraph says how the work differs, not only that other work exists [NN].

### 8. Limitations and conclusion

- Limitations go in the main text, not the appendix [NN, SPJ: acknowledge weaknesses].
- Say what is open and what is settled [LS §5.1].
- Conclusion: CL wants one short paragraph that makes the claims concrete with numbers; NN says skip it. In a short paper, fold it into the discussion.

### 9. Prose

- Active voice: "we ran", not "experiments were run" [SPJ].
- Plain words: "find out", not "ascertain"; "yearly", not "on an annual basis" [SPJ].
- Short sentences, one idea each [NN].
- Define each term once, before first use; then use only that term. No synonyms for one object [CL].
- No bare "this" or "it": write "this rise", "this check" [CL].
- Never "for various reasons"; give the reasons. No "etc." unless the list is obvious [CL].
- Numbers with units and uncertainty: "12.3% (95% CI 10.1–14.5), n = 4,210" [NN; Figure guide §7].

### 10. Revision

- Write in this order: claims, abstract, bullet outline, introduction, draft figures, full draft; get feedback after each step [NN].
- A reader can read a draft fresh only once. Ask "where did you get lost?", not for typos [SPJ].
- Read as a hostile reviewer: for each claim, find the evidence, then the easiest objection [NN, SPJ].
- Treat every critical comment as a request to make the text clearer, not as an error by the reader [SPJ].
- Send the draft to the authors of the closest prior work and ask if you describe it fairly [SPJ].
- Print the final PDF and read it on paper; run a spell checker [CL, SPJ].
- Keep the lab notebook current: every number in the paper traces to a dated entry and a script [JS].

### Pre-submission checklist

1. Each core claim is one sentence and appears in the abstract, the introduction and the section headings.
2. Each contribution bullet is refutable and has a section and figure reference.
3. The abstract names the genre, the data size and one number per claim.
4. Every claim in the introduction has evidence in the body.
5. Every speculation is labeled and sits in one marked paragraph.
6. Each term is defined once; no synonyms for one object.
7. Every estimate has n and a 95% CI from the correct unit.
8. Each main result has a strong baseline next to it.
9. Robustness checks are listed in one sentence in the main text and tabled in the appendix.
10. Limitations appear in the main text.
11. Prompts, model versions, thresholds and data dates are in the appendix; the code link works.
12. No passive voice in results sentences; no bare "this"; no "etc."; no "various reasons".
13. No roadmap paragraph.
14. Related work says how each cited line of work differs.
15. At least two readers outside the project read the introduction and said where they got lost.
16. Figures pass the Figure guide checklist.
17. Spell checked; printed and read once.
18. Within the page limit plus references and appendix; submitted before the deadline [SPJ].
19. Numbers in the paper come from files in `reports/` or `paper/tables/`; none are typed by hand.

## Figure guide

Sources:
- Rougier, Droettboom, Bourne. *Ten Simple Rules for Better Figures.* PLoS Comput Biol 10(9), 2014 (https://osf.io/download/3mg6s/).
- Zitnik. *Effective Use of Figures in Research Papers.* Stanford lecture, 2019 (https://cs.stanford.edu/~marinka/slides/marinka-figures19.pdf).

Each rule below cites its source: R = Rougier et al. rule number, Z = Zitnik principle or slide.

### 1. One message per figure (R2, Z: "focus on key information")

- Write the message as one sentence before drawing. The caption opens with that sentence.
- Material that does not serve the message moves to an appendix figure or table (Z: supplementary figures hold robustness).

### 2. Structure (Z: visual structure of a paper)

- Figure 1 answers the paper's main question (Z: "Figure 1 answers the question asked by the title").
- Later figures unpack the key result, then give orthogonal evidence.
- Figure order follows the order of the research questions. Each figure sits near the text that cites it.

### 3. Audience and medium (R1, R3)

- Know who reads the figure and on what medium. Readers of a PDF can look a long time, so details may stay, but the main message must be visible at first glance. Slides need fewer details, thicker lines and bigger text.
- Draw at final size (e.g. one column or full text width of the template). Never scale in LaTeX, so fonts keep their size.
- Text at 7–8 pt, never below 6 pt. Lines 1.0–1.5 pt, markers 3–4 pt.

### 4. Captions are not optional (R4)

- First sentence: the message. Then: what each mark is, what the error bars are, the sample, and which quantity (raw or adjusted).
- A caption must let a reader understand the figure without the body text.

### 5. Do not trust defaults (R5)

- Set fonts, sizes, ticks, colors and spacing by hand in one shared style block.
- Use the document's font family so figure text matches body text.
- Few ticks (3–5 per axis), short tick labels.

### 6. Color with a purpose (R6, Z: Brewer palettes, color blindness)

- If an element has no reason to be colored, make it black or grey.
- Use one colorblind-safe qualitative palette (e.g. Okabe–Ito) for the whole paper. A color keeps one meaning in every figure; write the color map down before drawing.
- Sequential data get one hue (light to dark); signed changes get a diverging pair with a neutral middle.
- Never use color alone to carry identity: add a direct label or a marker shape.

### 7. Do not mislead (R7)

- Same quantity, same axis range across panels that are compared.
- Mark a key event the same way in every time plot.
- Say in the axis label or caption whether a value is raw or adjusted.
- No dual y-axes, no 3-D, no pie charts.

### 8. Avoid chartjunk (R8, Z: Tufte data-ink ratio)

- No top and right spines, no frames on legends, no background colors except a band that marks a key period.
- Grid lines only when the reader must read values off them; then light grey.
- Label lines directly at their ends instead of a legend box when there are 4 or fewer series.

### 9. Message trumps beauty (R9)

- Choose the simplest chart that carries the message: dot plots with intervals for estimates, lines for time, dumbbells for before and after.
- Put the comparison the reader must make side by side on a common axis.

### 10. Right tool, reproducible output (R10, Z: key rules)

- Generate all figures from scripts that read saved result files (CSV, parquet), never from notebooks or by hand.
- Save as vector PDF.
- Render every page to an image and look at it after each change: no overlaps, no clipped text, legends off the data.

### Checklist before shipping a figure

1. Message sentence written, and the caption starts with it.
2. Drawn at final width; smallest text ≥ 6 pt.
3. Colors follow the paper's color map; nothing colored without a reason.
4. No label, legend or marker overlaps data.
5. Rendered page checked by eye.
