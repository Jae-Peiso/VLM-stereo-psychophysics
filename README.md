# VLM Stereo Psychophysics

**Question:** when a vision-language model (VLM) receives a stereo pair, does it use binocular disparity to judge depth, or does it answer from monocular cues and response bias?

This project applies methods from human psychophysics to vision-language models. The stimuli are random-dot stereograms: each image alone is uniform noise, and a central square exists only in the disparity between the two images. A model that reports the square's depth correctly has to match points across the two images (the correspondence problem). Controls separate that ability from response bias (zero disparity) and from accidental monocular cues (one image at a time).

## Status

In progress (November 2026).

| Component | Status |
| --- | --- |
| Random-dot stereogram generator (`src/vlm_stereo/stimuli.py`) | Done, validated numerically |
| Design decisions log ([DECISIONS.md](DECISIONS.md)) | Ongoing |
| Local inference with Qwen2.5-VL-3B-Instruct (4-bit, MLX) | Running |
| pytest tests for the generator | Next |
| Backend with log-probability scoring, trial generator, SQLite results | Planned |
| Pilot, psychometric fits, signal detection analysis, figures | Planned |

## Design summary

- **Task:** single-interval yes/no ("Is the central square in front of or behind the background?"), analyzed with signal detection theory so that sensitivity and bias are measured separately.
- **Conditions:** signed disparities (0, ±2, ±4, ±8, ±16 px), counterbalanced answer order, and a monocular control.
- **Response measure:** log-probabilities of "front" and "behind", giving a graded decision variable on every trial.
- **Stimulus size:** chosen so the model's preprocessing does not resample the images, and verified from the processor's own output.

The reasoning behind each choice, including the alternatives considered, is in [DECISIONS.md](DECISIONS.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Inference uses MLX, which runs on Apple silicon Macs.

```
git clone https://github.com/Jae-Peiso/VLM-stereo-psychophysics.git
cd VLM-stereo-psychophysics
uv sync
```

Generate a stereogram pair:

```python
from vlm_stereo.stimuli import make_rds

left, right = make_rds(disparity=8, seed=0)   # uint8 arrays, values 0 or 255
```

Positive disparity is crossed (square in front). With the left image displayed on the left, cross-eyed free-fusion shows a positive-disparity square behind the background, since cross-fusing swaps the eyes.

## Repository layout

```
src/vlm_stereo/   package code (stimuli.py: stereogram generator)
notebooks/        exploration
DECISIONS.md      design decisions and their rationale
```

## License

MIT. See [LICENSE](LICENSE).
