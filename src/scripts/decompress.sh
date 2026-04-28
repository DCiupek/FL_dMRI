#!/bin/bash

DATASET="${1:-}"

usage () {
  echo "Usage: $0 DATASET_PATH.pkl"
  echo ""
  echo "    Decompress the gzipped nibabel files to speed up laoding. It will generate the new .pkl file"
  echo "    with paths appropriately changed with '_decompressed' appended before the '.pkl'."
}

if [[ -z "$DATASET" ]]; then
  echo "$0: no dataset" 1>&2
  usage
  exit 2
fi
DATASET=$(realpath "${DATASET}")
if [[ ! -f "${DATASET}" ]]; then
  echo "$0: no such file '${DATASET}'" 1>&2
  exit 1
fi

PATHS_CODE="import pickle
with open('${DATASET}', 'rb') as f:
    dd = pickle.load(f)
for v in dd['data_path']:
    print(v)
"

decompressed=()
for path in $(python3 -c "${PATHS_CODE}");
do
    if [[ "${path}" == *.gz ]]; then

        if ! gzip -d "${path}"; then
            exit 1
        fi
    fi
    echo "${path%.gz}"
    decompressed+=("${path%.gz}")
done

python3 -c "import pickle
import sys
with open('${DATASET}', 'rb') as f:
    dd = pickle.load(f)
dd['data_path'] = sys.argv[1:]
with open('${DATASET%.pkl}_decompressed.pkl', 'wb') as f:
    pickle.dump(dd, f)
" "${decompressed[@]}"
