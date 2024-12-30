#!/bin/bash

set -exo pipefail

if [[ -z "$1" ]]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

VERSION_TO_REPLACE="0.0.0-dev"
VERSION_TAG=$1
VERSION_TARGET=${VERSION_TAG#v}

sed -i "s/$VERSION_TO_REPLACE/$VERSION_TARGET/" pyproject.toml
sed -i "s/$VERSION_TO_REPLACE/$VERSION_TARGET/" rust/Cargo.toml
