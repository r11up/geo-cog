"""Bundle for GPU vision models (Urban Appraisal, public data): the stimulus images, per-person
ratings (recoded, 62 participants) and scene features. Written to derived/kaggle_scenes/."""
import json
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import pandas as pd  # noqa: E402

from urbeeg.config import DERIVED, STIMULI_DIRS  # noqa: E402
from urbeeg.dataset import processed_participants, ratings_wide  # noqa: E402


def main(dataset_id: str):
    out = DERIVED / "kaggle_scenes"
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    for d in STIMULI_DIRS:
        for f in sorted(Path(d).glob("*.jpg")):
            shutil.copy2(f, out / "images" / f.name)
            n += 1
    ratings_wide(processed_participants()).reset_index().to_csv(out / "ratings_person.csv", index=False)
    shutil.copy2(DERIVED / "kaggle_bundle" / "scene_features.csv", out / "scene_features.csv")
    json.dump({"title": "Urban Appraisal scenes and ratings", "id": dataset_id,
               "licenses": [{"name": "CC-BY-4.0"}],
               "description": "Stimulus images (BeMoBIL urban_appraisal-experiment, BSD-3), per-person ratings "
                              "derived from OpenNeuro ds006850, and scene features. Private working copy."},
              open(out / "dataset-metadata.json", "w"), indent=1)
    print(f"{n} images, ratings and scene features in {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "utsavghhjg/urban-appraisal-scenes")
