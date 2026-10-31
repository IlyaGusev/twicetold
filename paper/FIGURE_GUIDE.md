# Figure guide

Sources:
- Rougier, Droettboom, Bourne. *Ten Simple Rules for Better Figures.* PLoS Comput Biol 10(9), 2014 (https://osf.io/download/3mg6s/).
- Zitnik. *Effective Use of Figures in Research Papers.* Stanford lecture, 2019 (https://cs.stanford.edu/~marinka/slides/marinka-figures19.pdf).

Each rule below cites its source: R = Rougier et al. rule number, Z = Zitnik principle or slide.

## 1. One message per figure (R2, Z: "focus on key information")

- Write the message as one sentence before drawing. The caption opens with that sentence.
- Material that does not serve the message moves to an appendix figure or table (Z: supplementary figures hold robustness).

## 2. Structure (Z: visual structure of a paper)

- Figure 1 answers the paper's main question (Z: "Figure 1 answers the question asked by the title").
- Later figures unpack the key result, then give orthogonal evidence.
- Figure order follows the order of the research questions. Each figure sits near the text that cites it.

## 3. Audience and medium (R1, R3)

- Know who reads the figure and on what medium. Readers of a PDF can look a long time, so details may stay, but the main message must be visible at first glance. Slides need fewer details, thicker lines and bigger text.
- Draw at final size (e.g. one column or full text width of the template). Never scale in LaTeX, so fonts keep their size.
- Text at 7–8 pt, never below 6 pt. Lines 1.0–1.5 pt, markers 3–4 pt.

## 4. Captions are not optional (R4)

- First sentence: the message. Then: what each mark is, what the error bars are, the sample, and which quantity (raw or adjusted).
- A caption must let a reader understand the figure without the body text.

## 5. Do not trust defaults (R5)

- Set fonts, sizes, ticks, colors and spacing by hand in one shared style block.
- Use the document's font family so figure text matches body text.
- Few ticks (3–5 per axis), short tick labels.

## 6. Color with a purpose (R6, Z: Brewer palettes, color blindness)

- If an element has no reason to be colored, make it black or grey.
- Use one colorblind-safe qualitative palette (e.g. Okabe–Ito) for the whole paper. A color keeps one meaning in every figure; write the color map down before drawing.
- Sequential data get one hue (light to dark); signed changes get a diverging pair with a neutral middle.
- Never use color alone to carry identity: add a direct label or a marker shape.

## 7. Do not mislead (R7)

- Same quantity, same axis range across panels that are compared.
- Mark a key event the same way in every time plot.
- Say in the axis label or caption whether a value is raw or adjusted.
- No dual y-axes, no 3-D, no pie charts.

## 8. Avoid chartjunk (R8, Z: Tufte data-ink ratio)

- No top and right spines, no frames on legends, no background colors except a band that marks a key period.
- Grid lines only when the reader must read values off them; then light grey.
- Label lines directly at their ends instead of a legend box when there are 4 or fewer series.

## 9. Message trumps beauty (R9)

- Choose the simplest chart that carries the message: dot plots with intervals for estimates, lines for time, dumbbells for before and after.
- Put the comparison the reader must make side by side on a common axis.

## 10. Right tool, reproducible output (R10, Z: key rules)

- Generate all figures from scripts that read saved result files (CSV, parquet), never from notebooks or by hand.
- Save as vector PDF.
- Render every page to an image and look at it after each change: no overlaps, no clipped text, legends off the data.

## Checklist before shipping a figure

1. Message sentence written, and the caption starts with it.
2. Drawn at final width; smallest text ≥ 6 pt.
3. Colors follow the paper's color map; nothing colored without a reason.
4. No label, legend or marker overlaps data.
5. Rendered page checked by eye.
