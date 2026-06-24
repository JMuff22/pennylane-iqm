#!/usr/bin/env bash
# Release helper invoked by .github/workflows/tag_and_release.yml.
# Subcommands:
#   verify_changelog_version  Ensure CHANGELOG.md has a fresh semver entry on top.
#   create_new_tag            Push a "vX.Y.Z" tag and create a GitHub release.
set -euo pipefail

CHANGELOG="${CHANGELOG:-CHANGELOG.md}"

top_version() {
	grep -m1 -E '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' "$CHANGELOG" \
		| sed -E 's/^## \[([0-9]+\.[0-9]+\.[0-9]+)\].*/\1/'
}

verify_changelog_version() {
	local v
	v=$(top_version || true)
	if [[ -z "${v}" ]]; then
		echo "::error::No semver entry found at top of ${CHANGELOG}" >&2
		exit 1
	fi
	echo "Top changelog version: ${v}"
	if git rev-parse "v${v}" >/dev/null 2>&1; then
		echo "::error::Tag v${v} already exists" >&2
		exit 1
	fi
}

create_new_tag() {
	local v
	v=$(top_version || true)
	if [[ -z "${v}" ]]; then
		echo "::error::No semver entry found at top of ${CHANGELOG}" >&2
		exit 1
	fi
	local tag="v${v}"
	echo "Creating tag ${tag}"
	git config user.email "github-actions[bot]@users.noreply.github.com"
	git config user.name "github-actions[bot]"
	git tag -a "${tag}" -m "Release ${tag}"
	git push origin "${tag}"
	gh release create "${tag}" --title "${tag}" --generate-notes
}

case "${1:-}" in
	verify_changelog_version) verify_changelog_version ;;
	create_new_tag) create_new_tag ;;
	*)
		echo "Usage: $0 {verify_changelog_version|create_new_tag}" >&2
		exit 2
		;;
esac
