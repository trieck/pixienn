# Prosopo

Prosopo is Pixienn's one-class person-detection dataset.  Every retained
ground-truth object has class `0`, named `person`; annotations for all other
object categories are discarded.

## Sources

The generated dataset is curated for the current CenterNet Prosopo training
configuration: each source image occupies a `384x384` Mosaic quadrant within
the `768x768` training input. Person boxes whose letterboxed width or height is
below 10 pixels at that effective per-source resolution are removed; images
with no usable boxes remaining are excluded. The source datasets are not
modified.

The generated dataset is an amalgamation of the locally configured splits:

| Source | Training images | Validation images | Person boxes |
| --- | ---: | ---: | ---: |
| COCO | 44,252 | 21,171 | 236,779 |
| KITTI | 1,060 | 263 | 3,178 |
| VOC | 3,718 | 1,027 | 10,437 |
| **Total** | **49,030** | **22,461** | **250,394** |

COCO `person` is mapped from class 0.  VOC `person` is mapped from class 14.
KITTI `pedestrian` and `person_sitting` are mapped to `person`.  KITTI
`cyclist` is intentionally excluded because its source box covers the
cyclist-and-bicycle object rather than a person-only extent.

The source train and validation partitions are preserved, and duplicate source
images are removed before writing the output lists.  The two generated lists
therefore contain disjoint image files.

## Layout

```text
prosopo/
├── images/          # person-containing source images
├── labels/          # YOLO labels; every row begins with class 0
├── train.txt
├── val.txt          # deterministic 1,000-image diagnostic validation set
├── prosopo.names
├── manifest.jsonl   # source-to-output provenance for every image
├── summary.json
└── build.py         # deterministic rebuild script
```

Images are hard-linked from the local source datasets whenever the filesystem
allows it; otherwise the builder copies them.  This keeps the output paths
under `resources/data/prosopo` while avoiding unnecessary duplication where
possible.

To rebuild after changing source lists or mappings:

```bash
python3 resources/data/prosopo/build.py
```

The generated lists can be used by a Pixienn configuration with:

```yaml
configuration:
  labels: ../data/prosopo/prosopo.names

training:
  train-images: ../data/prosopo/train.txt
  train-labels: ../data/prosopo/labels
  val-images: ../data/prosopo/val.txt
  val-labels: ../data/prosopo/labels
```

The configuration points to the deterministic 1,000-image validation set, so
routine validation evaluates a balanced sample and returns to training
quickly.  There is intentionally no full-validation list: the complete
held-out collection is retained only in the provenance manifest and is not an
available validation workload.
