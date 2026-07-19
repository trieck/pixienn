#!/usr/bin/env bash

set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: check-training-data.sh [--quick] CONFIG

Validate the model, labels, image manifests, and label directories referenced
by a PixieNN training configuration. Without --quick, every manifest entry and
its corresponding .txt label file are checked.
EOF
}

quick=false
config=""

while (($#)); do
    case "$1" in
        --quick)
            quick=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$config" ]]; then
                echo "Only one configuration may be checked." >&2
                exit 2
            fi
            config=$1
            ;;
    esac
    shift
done

if [[ -z "$config" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -f "$config" ]]; then
    echo "Configuration not found: $config" >&2
    exit 1
fi

config=$(realpath -- "$config")
config_dir=$(dirname -- "$config")

yaml_value()
{
    local key=$1
    awk -v key="$key" '
        $1 == key ":" {
            sub(/^[^:]+:[[:space:]]*/, "")
            sub(/[[:space:]]+#.*$/, "")
            gsub(/^"|"$/, "")
            print
            exit
        }
    ' "$config"
}

resolve_from()
{
    local base=$1
    local value=$2
    if [[ "$value" = /* ]]; then
        realpath -m -- "$value"
    else
        realpath -m -- "$base/$value"
    fi
}

require_file()
{
    local description=$1
    local path=$2
    if [[ ! -f "$path" ]]; then
        echo "Missing $description: $path" >&2
        return 1
    fi
    printf '  %-18s %s\n' "$description:" "$path"
}

require_directory()
{
    local description=$1
    local path=$2
    if [[ ! -d "$path" ]]; then
        echo "Missing $description: $path" >&2
        return 1
    fi
    printf '  %-18s %s\n' "$description:" "$path"
}

model_value=$(yaml_value model)
labels_value=$(yaml_value labels)
train_images_value=$(yaml_value train-images)
train_labels_value=$(yaml_value train-labels)
val_images_value=$(yaml_value val-images)
val_labels_value=$(yaml_value val-labels)

for value_name in model_value labels_value train_images_value train_labels_value val_images_value val_labels_value; do
    if [[ -z "${!value_name}" ]]; then
        echo "Required training key is missing in $config: ${value_name%_value}" >&2
        exit 1
    fi
done

model=$(resolve_from "$config_dir" "$model_value")
labels=$(resolve_from "$config_dir" "$labels_value")
train_images=$(resolve_from "$config_dir" "$train_images_value")
train_labels=$(resolve_from "$config_dir" "$train_labels_value")
val_images=$(resolve_from "$config_dir" "$val_images_value")
val_labels=$(resolve_from "$config_dir" "$val_labels_value")

echo "Checking $config"
require_file "model" "$model"
require_file "class labels" "$labels"
require_file "train manifest" "$train_images"
require_directory "train labels" "$train_labels"
require_file "val manifest" "$val_images"
require_directory "val labels" "$val_labels"

manifest_entries()
{
    awk 'NF && $1 !~ /^#/ { count++ } END { print count + 0 }' "$1"
}

model_scalar()
{
    local key=$1
    awk -v key="$key" '$1 == key ":" { print $2; exit }' "$model"
}

validation_scalar()
{
    local key=$1
    awk -v key="$key" '
        $1 == "validation:" { in_validation=1; next }
        in_validation && $1 == key ":" { print $2; exit }
        in_validation && /^[[:space:]]{2}[a-zA-Z0-9_-]+:/ { exit }
    ' "$model"
}

batch=$(model_scalar batch)
subdivisions=$(model_scalar subdivisions)
subdivisions=${subdivisions:-1}
validation_enabled=$(validation_scalar enabled)
validation_interval=$(validation_scalar interval)
validation_batches=$(validation_scalar batches)
train_entries=$(manifest_entries "$train_images")
val_entries=$(manifest_entries "$val_images")

if [[ "$batch" =~ ^[0-9]+$ && "$subdivisions" =~ ^[0-9]+$ && subdivisions -gt 0 ]]; then
    loader_batch=$((batch / subdivisions))
    if ((loader_batch > 0)); then
        expected_train_batches=$(((train_entries + loader_batch - 1) / loader_batch))
        expected_val_batches=$(((val_entries + loader_batch - 1) / loader_batch))
        printf '  %-18s %d train, %d validation\n' "manifest sizes:" "$train_entries" "$val_entries"
        printf '  %-18s %d images (%d subdivision%s)\n' \
            "effective batch:" "$loader_batch" "$subdivisions" "$([[ "$subdivisions" == 1 ]] && echo "" || echo "s")"

        if [[ "${validation_enabled,,}" == true ]]; then
            if [[ "$validation_interval" =~ ^[0-9]+$ ]] && \
               ((validation_interval > expected_train_batches * 2 || validation_interval * 2 < expected_train_batches)); then
                echo "Warning: validation interval $validation_interval differs substantially from one manifest pass ($expected_train_batches batches)." >&2
            fi
            if [[ "$validation_batches" =~ ^[0-9]+$ ]] && ((validation_batches > expected_val_batches)); then
                echo "Warning: validation requests $validation_batches batches, but the manifest contains $expected_val_batches; PixieNN will clamp to the available data." >&2
            elif [[ "$validation_batches" =~ ^[0-9]+$ ]] && ((validation_batches < expected_val_batches)); then
                echo "Warning: validation covers only $validation_batches of $expected_val_batches available batches." >&2
            fi
        fi
    fi
fi

if $quick; then
    echo "Quick preflight passed."
    exit 0
fi

check_manifest()
{
    local description=$1
    local manifest=$2
    local label_dir=$3
    local manifest_dir
    manifest_dir=$(dirname -- "$manifest")

    local total=0
    local missing_images=0
    local missing_labels=0
    local duplicates=0
    local reported=0
    local line image image_name label
    declare -A seen=()

    while IFS= read -r line || [[ -n "$line" ]]; do
        line=${line//$'\r'/}
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "$line" == \#* ]] && continue

        ((total += 1))
        if [[ "$line" = /* ]]; then
            image=$line
        else
            image="$manifest_dir/$line"
        fi
        if [[ -n "${seen[$image]+present}" ]]; then
            ((duplicates += 1))
        else
            seen[$image]=1
        fi

        if [[ ! -f "$image" ]]; then
            ((missing_images += 1))
            if ((reported < 20)); then
                echo "  missing image: $image" >&2
                ((reported += 1))
            fi
        fi

        image_name=${image##*/}
        label="$label_dir/${image_name%.*}.txt"
        if [[ ! -f "$label" ]]; then
            ((missing_labels += 1))
            if ((reported < 20)); then
                echo "  missing label: $label" >&2
                ((reported += 1))
            fi
        fi
    done < "$manifest"

    if ((total == 0)); then
        echo "$description manifest is empty: $manifest" >&2
        return 1
    fi

    printf '  %-18s %d entries, %d duplicates, %d missing images, %d missing labels\n' \
        "$description:" "$total" "$duplicates" "$missing_images" "$missing_labels"

    if ((duplicates || missing_images || missing_labels)); then
        return 1
    fi
}

check_manifest "train entries" "$train_images" "$train_labels"
check_manifest "val entries" "$val_images" "$val_labels"
echo "Full data verification passed."
