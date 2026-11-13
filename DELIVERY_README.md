# RMAL-CRNN ICLR soundness revision

This directory is the revised `main2` submission source and response package.

## Contents

- `main2.tex` and `main2.pdf`: revised manuscript and compiled PDF.
- `sections/`, `appendix/`, `figures/`, `math_commands.tex`, `iclr2027_conference.sty`, `iclr2027_conference.bst`, and `references.bib`: complete LaTeX source.
- `code/`: retained experiment and analysis code, excluding Python bytecode caches.
- `formula_checks.json`: forward-operator checks against the supplied PyTorch implementations.
- `rebuttal.docx`: concise English response to the reviewer.

## Revisions

The equations match the implemented GST, LMA, FNSAR, and CRNN operators. The MAtt-A row follows Pan, Chou, and Wei (NeurIPS 2022), and MambaSL and TimeMixer++ cite their formal method publications. The Kielce dataset is linked to Zenodo record 15190811, with the ten selected UAV classes mapped to released folders and model names in Appendix A. Seed language is `multi-seed`, and Codex assistance is disclosed in the Statements section.

Reported table values and figure assets are unchanged. The manuscript compiles without undefined citations, undefined references, or overfull boxes in the final log. The appendix records the shared multi-seed protocol and model configurations.
