# Prosopo

Prosopo is Pixienn's one-class person-detection dataset.  Every retained
ground-truth object has class `0`, named `person`; annotations for all other
object categories are discarded.

## Sources

The generated dataset is an amalgamation of the locally configured splits:

| Source | Training images | Validation images | Person boxes |
| --- | ---: | ---: | ---: |
| COCO | 45,174 | 21,634 | 276,638 |
| KITTI | 1,446 | 350 | 4,709 |
| VOC | 3,723 | 1,027 | 10,456 |
| **Total** | **50,343** | **23,011** | **291,803** |

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
