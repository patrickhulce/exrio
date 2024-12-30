#!/bin/bash

set -exo pipefail

if [[ -z "$1" ]]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

VERSION_TO_REPLACE="0.0.0-dev"
VERSION_TAG=$1
VERSION_TARGET=${VERSION_TAG#v}

OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    SED_OPTIONS=("-i" "")
else
    SED_OPTIONS=("-i")
fi

sed "${SED_OPTIONS[@]}" "s/$VERSION_TO_REPLACE/$VERSION_TARGET/" pyproject.toml
sed "${SED_OPTIONS[@]}" "s/$VERSION_TO_REPLACE/$VERSION_TARGET/" rust/Cargo.toml

# Confirm the version has been updated.
grep "version =" pyproject.toml
grep "version = \"$VERSION_TARGET\"" pyproject.toml
grep "version =" rust/Cargo.toml
grep "version = \"$VERSION_TARGET\"" rust/Cargo.toml
