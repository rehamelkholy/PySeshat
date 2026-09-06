# `CITATION.cff` completion checklist

`CITATION.cff` at the repo root is filled in with real values everywhere
the repository itself supplies them (title, authors, license, GitHub
URL, version). A few fields cannot be known yet and are left as
clearly-marked placeholders, kept schema-valid (`cffconvert --validate`
passes as-is) so GitHub's "Cite this repository" widget doesn't choke on
the file in the meantime. Complete these by hand as each becomes true:

- **Your ORCID iD** — not present at all yet (an invalid placeholder
  would fail CFF's pattern validation for this field, so it's omitted
  rather than faked). Once you have one
  (register at <https://orcid.org/register>), add to both `authors` and
  `preferred-citation.authors`:
  ```yaml
  orcid: "https://orcid.org/0000-0000-0000-0000"
  ```
- **`date-released`** — placeholder `"9999-12-31"`. Set to the actual
  date (YYYY-MM-DD) when you cut the GitHub release.
- **`identifiers[0].value`** (Zenodo DOI) — placeholder
  `"10.0000/TODO-ZENODO-DOI"` (`10.0000` is not a real DOI prefix, so
  this doesn't resolve). A real Zenodo DOI is minted automatically the
  first time you enable Zenodo-GitHub integration
  (<https://zenodo.org/account/settings/github/>) and then cut a GitHub
  release — Zenodo will show you the DOI it assigned; paste it in here.
- **`preferred-citation.identifiers[0].value`** (arXiv ID) — placeholder
  `"arXiv:TODO-ARXIV-ID"`. Replace once you submit the preprint, e.g.
  `"arXiv:2601.12345"`.
- **`preferred-citation.title`** — placeholder `"TODO-PAPER-TITLE"`.
  Set to the paper's final title.
- **`preferred-citation.volume`**, **`.pages`**, **`.year`** —
  placeholders (`"TODO-VOLUME"`, `"TODO-PAGES"`, `9999`). Fill in once
  Astronomy & Astrophysics accepts the paper and assigns these.

Once all of the above are real, `pip install cffconvert && cffconvert
--validate -i CITATION.cff` re-confirms the file is well-formed.

The same reminders also live as comments directly inside `CITATION.cff`
itself, so they travel with the file even if this note is moved or
deleted.
