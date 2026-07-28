#!/usr/bin/env bash
# Build (or refresh) the `anon` branch: an author-anonymous copy of this repo suitable for
# serving through anonymous.4open.science during ARR review.
#
# ARR forbids linking an authored remote. Beyond the remote URL, this repository leaks the
# author's name through absolute paths baked into run artifacts by lm-eval and LightCompress:
# ~105 tracked files contain `/mnt/d/Abrar/SEQ/seq_v4`, and 11 lm-eval output directories are
# named after it (`__mnt__d__Abrar__SEQ__seq_v4__...`). Serving the repo unscrubbed would
# de-anonymise the submission on the first directory listing.
#
# What this does NOT do: rewrite git history. anonymous.4open.science serves a file tree, not
# a commit log, so scrubbing names and contents is what matters. If you ever publish the branch
# to a real remote instead, squash it first.
#
#   bash scripts/make_anon_mirror.sh          # build/refresh the branch, then verify
#   bash scripts/make_anon_mirror.sh --check  # verify only, no changes
#
set -euo pipefail

REAL="/mnt/d/Abrar/SEQ/seq_v4"      # the authored absolute path baked into artifacts
FAKE="/work/seq_v4"                 # its neutral replacement
REAL_DIR="__mnt__d__Abrar__SEQ__seq_v4__"
FAKE_DIR="__work__seq_v4__"
HOME_REAL="/home/abrar"
HOME_FAKE="/home/user"
BRANCH="anon"

cd "$(git rev-parse --show-toplevel)"

# Patterns that must not survive anywhere in the tree. Kept separate from the substitutions
# above so a new leak (a different username, a machine name) fails the check loudly rather
# than being silently missed.
LEAKS=(Abrar abrar SEQQQ farzana AbrarUI12)

verify() {
  local bad=0
  for pat in "${LEAKS[@]}"; do
    # -I skips binary files; the PDF is a build artifact and is checked separately.
    if git grep -I -l -i -- "$pat" >/dev/null 2>&1; then
      echo "LEAK: '$pat' still present in:"
      git grep -I -l -i -- "$pat" | sed 's/^/    /'
      bad=1
    fi
  done
  if find . -path ./.git -prune -o -iname '*abrar*' -print | grep -q .; then
    echo "LEAK: paths still named after the author:"
    find . -path ./.git -prune -o -iname '*abrar*' -print | sed 's/^/    /'
    bad=1
  fi
  if [[ $bad -eq 0 ]]; then echo "CLEAN: no author identifiers in tree or path names."; fi
  return $bad
}

if [[ "${1:-}" == "--check" ]]; then verify; exit $?; fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty. Commit or stash first." >&2; exit 1
fi

START="$(git rev-parse --abbrev-ref HEAD)"
git branch -f "$BRANCH" HEAD
git checkout "$BRANCH"

# ---- 1. rename the lm-eval output directories -------------------------------------------- #
# Deepest first, so renaming a parent never invalidates a queued child path.
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  new="$(dirname "$d")/$(basename "$d" | sed "s|$REAL_DIR|$FAKE_DIR|")"
  [[ "$d" == "$new" ]] && continue
  git mv "$d" "$new"
  echo "renamed: $(basename "$d") -> $(basename "$new")"
done < <(find . -path ./.git -prune -o -type d -name "${REAL_DIR}*" -print | sort -r)

# ---- 2. rewrite the absolute paths inside tracked text files ------------------------------ #
mapfile -t FILES < <(git grep -I -l -e "$REAL" -e "$HOME_REAL" || true)
if [[ ${#FILES[@]} -gt 0 ]]; then
  for f in "${FILES[@]}"; do
    sed -i "s|$REAL|$FAKE|g; s|$HOME_REAL|$HOME_FAKE|g" "$f"
  done
  echo "rewrote paths in ${#FILES[@]} files"
fi

# ---- 3. drop the docs that exist only to drive our own machines --------------------------- #
# They are full of box-specific paths and say nothing a reviewer needs.
git rm -q --ignore-unmatch docs/RESEARCH_PC_RUNLIST.md docs/CODEX_PROMPT_JOB3.md \
                           docs/GPU_RUNBOOK_TODAY.md || true

git commit -q -am "Anonymised mirror: neutral paths, no author identifiers" || true

echo
verify || { echo; echo "Verification FAILED. Fix the leaks above before publishing."; exit 1; }

cat <<EOF

Branch '$BRANCH' is ready.

  1. Push it:            git push -f origin $BRANCH
  2. Register the repo at https://anonymous.4open.science, pointing at branch '$BRANCH'
  3. Open the anonymous link and spot-check a couple of runs/final/downstream/** paths
  4. Put the link in the camera-ready only -- see the ANONYMOUS marker in paper/main.tex

You are on '$BRANCH'. Return with:  git checkout $START
EOF
