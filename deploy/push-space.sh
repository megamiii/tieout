#!/usr/bin/env sh
# Push the current commit to a Hugging Face Space (Docker SDK).
#
#   HF_SPACE=your-username/tieout sh deploy/push-space.sh
#
# A Space reads its settings from YAML front matter in README.md, which GitHub
# would render as a table. So the push goes from a temporary worktree where
# README.md is replaced by deploy/huggingface/README.md; the GitHub README and
# the current branch are left untouched. Git asks for your Hugging Face
# username and an access token with write permission.
set -eu
: "${HF_SPACE:?set HF_SPACE to your-username/space-name}"

tmp="$(mktemp -d)"
cleanup() { git worktree remove --force "$tmp" 2>/dev/null || true; }
trap cleanup EXIT

git worktree add --quiet --detach "$tmp" HEAD
cp deploy/huggingface/README.md "$tmp/README.md"
cd "$tmp"
git add README.md
git commit --quiet -m "Hugging Face Space configuration"
if [ "${DRY_RUN:-0}" = "1" ]; then
  git log --oneline -2
  head -9 README.md
  exit 0
fi
git push --force "https://huggingface.co/spaces/$HF_SPACE" HEAD:main
