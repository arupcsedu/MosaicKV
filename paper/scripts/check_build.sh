#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for name in main thesis supplement; do
  test -s "build/${name}.pdf"
  if rg -n 'Undefined control sequence|LaTeX Error|AUTHORERR|Citation .* undefined|Reference .* undefined|There were undefined references|multiply defined|Overfull|destination with the same identifier' "build/${name}.log"; then
    printf 'Unresolved LaTeX issue in %s\n' "${name}" >&2
    exit 1
  fi
  if rg -n "Warning--|I couldn't open" "build/${name}.blg"; then
    exit 1
  fi
  if command -v pdftotext >/dev/null 2>&1; then
    if pdftotext "build/${name}.pdf" - | rg -n 'AUTHORERR'; then
      printf 'Template author metadata error in %s\n' "${name}" >&2
      exit 1
    fi
  fi
done
if rg -n 'and others' references.bib; then
  printf '%s\n' 'MLSys 2027 requires all authors in each bibliography entry.' >&2
  exit 1
fi
MAIN_PAGE=$(sed -n 's/.*\\newlabel{main-text-end}{{[^}]*}{\([0-9][0-9]*\)}.*/\1/p' build/main.aux)
if [[ -z "${MAIN_PAGE}" || "${MAIN_PAGE}" -gt 10 ]]; then
  printf 'Main text page limit check failed: %s (maximum 10)\n' "${MAIN_PAGE:-unknown}" >&2
  exit 1
fi
printf 'All builds passed citation/layout checks; main text ends on page %s.\n' "${MAIN_PAGE}"
