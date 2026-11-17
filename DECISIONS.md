# Design Decisions

Each entry records what was decided, why, and what was rejected. "In plain words" is the one-sentence version.

---

## D001: Task is yes/no, not 2AFC
Date: 2026-11-11
Decision: One stereogram per trial, two response labels ("front", "behind"). Analyze with yes/no signal detection: d′ = z(H) − z(F), c = −½[z(H) + z(F)].
Why: Measuring response bias is a goal. In a yes/no design the criterion is meaningful; true 2AFC largely removes it.
Alternatives: True 2AFC (two stereograms per trial, "which square is closer?"). Deferred to future work.
In plain words: Bias is part of the question, so I used a design where bias is measurable.

## D002: Even disparities only
Date: 2026-11-11
Decision: Disparity d must be even. Each eye shifts by d/2 pixels. The generator raises an error on odd d. Pilot set: 0, ±2, ±4, ±8, ±16 px. After the pilot, add points near the psychometric transition, symmetrically on both sides of zero.
Why: Odd d would require half-pixel shifts. Even d keeps every dot on integer pixel positions, so images stay strictly binary.
Alternatives: Asymmetric split (ceil/floor), or shifting one image by the full d.
In plain words: Even disparities keep every shift a whole number of pixels.

## D003: Sign convention
Date: 2026-11-11
Decision: Positive d = crossed disparity = "front". The square shifts right in the left-eye image and left in the right-eye image. Prompts state explicitly which image is which eye.
Why: Free-fusing reverses with technique (parallel vs. cross-eyed), and the model has no way to know which image is which eye unless told.
Validation note: Stimuli validated by cross-eyed free-fusion. With the left-eye image displayed on the left, cross-fusion swaps the eyes, so crossed disparity (+d) correctly appears behind. Sign also confirmed numerically: in the square's interior, left[x] == right[x - d].
In plain words: The sign is fixed in code and stated in the prompt, so nothing depends on guessing.

## D004: Both disparity signs are required
Date: 2026-11-11
Decision: Test crossed and uncrossed disparities symmetrically.
Why: With one sign only, a model that always answers "front" scores perfectly, and bias can't be separated from sensitivity. The psychometric midpoint gives bias; the slope gives sensitivity. Uncrossed disparity is common in natural vision; it is unrelated to wall-eyed free-fusing.
In plain words: Testing both signs separates seeing depth from preferring an answer.

## D005: RDS construction
Date: 2026-11-11
Decision: Scattered square dots at integer pixel positions. Square membership is decided by dot center; whole dots are drawn, never clipped at the square boundary. For each eye: remove background dots under the shifted square (occlusion), add the square's dots, and refill the vacated strip with fresh dots generated independently in each eye. Output uint8 arrays with values 0 and 255.
Why: The square must be invisible in each image alone. A blank strip, a double-density strip, a grid phase break, or a line of clipped dots would each leak a monocular cue. Occluding instead of pushing keeps density uniform and avoids giving the background its own disparity.
Alternatives: Grid of dot cells. Rejected because d/2 not a multiple of dot size creates a visible grid phase break at the square's edges.
Check: Dot density in vertical strips should be flat across each image.
In plain words: Every region of each image has the same statistics, so the square exists only in the disparity.

## D006: Generator is numpy; Pillow only saves
Date: 2026-11-11
Decision: Generate with numpy. Save with Pillow as lossless 8-bit grayscale PNG (mode "L"). Never JPEG. No OpenCV.
Why: Pure-numpy generation is easy to test. Pillow is already a dependency of mlx-vlm. JPEG would add artifacts around every dot. OpenCV adds a large dependency with nothing needed here.
In plain words: Numpy builds the stimuli, Pillow writes lossless files.

## D007: Generator is model-agnostic; stimulus size is chosen per model
Date: 2026-11-11
Decision: make_rds takes image_size as a plain parameter. The trial config chooses a size the model won't resample: for Qwen2.5-VL, multiples of 28 px (14 px patches merged 2×2) within its pixel limits, e.g. 448 × 448. Log the processor's actual patch grid (image_grid_thw) with every run.
Why: Resampling would blur dots and change disparities by fractions of a pixel. Other models have different preprocessing (some resize to a fixed square), so the rule belongs in config, not the generator. Checking the processor's output proves the model saw the stimulus unaltered.
In plain words: Stimuli are sized so the model never resizes them, and I verify that from the model's own preprocessing.

## D008: Focus on the correspondence problem; resolution is a stated limitation
Date: 2026-11-11
Decision: Frame the question as whether the model matches patches across the two images. Report the patch/resolution issue as a limitation, without claiming sub-patch disparities are destroyed.
Why: Patch embeddings are linear over pixels, so small shifts still change the embedding. The real question is cross-image matching.
In plain words: I controlled resolution, but the core question is correspondence.

## D009: Response measure is log-probabilities, with greedy decoding
Date: 2026-11-11
Decision: Record the log-probabilities of "front" and "behind" on every trial, plus the generated text. Temperature 0.
Why: Greedy decoding is deterministic, so variability comes only from dot patterns. Log-probabilities give a graded decision variable per trial, closer to an SDT model and more efficient than binary answers.
In plain words: The model's relative confidence in each answer is more informative than the answer alone.

## D010: Counterbalance answer order; shuffle trial order
Date: 2026-11-11
Decision: Counterbalance the order of "front" and "behind" in the prompt (crossed as a factor). Shuffle trial order within a run.
Why: Small models have position biases in answer options. Trial order doesn't affect a stateless model, but shuffling protects the dataset if a run dies partway through.
In plain words: Answer order is counterbalanced because it matters; trial order is shuffled because it's cheap insurance.

## D011: Cache key includes the stimulus
Date: 2026-11-11
Decision: The cache key hashes model, prompt, generation settings, image bytes, image order, and preprocessing settings.
Why: A key without the image would give every seed at the same disparity one cached answer, silently faking a perfectly consistent model.
In plain words: Each stimulus gets its own cache entry, so caching can't fake a result.
Date: 2026-11-16
Observation: The first couple of runs of probe_logprobs.py with slightly different prompt (missing spaces etc.) but the same seed produced different log-probabilities. **This means that the prompt's wording itself is an experimental condition.** 

## D012: Seeds
Date: 2026-11-11
Decision: make_rds uses a local np.random.default_rng(seed); default None for exploration. Experiments pass explicit integer seeds. The config holds one master seed; the trial generator derives one seed per dot pattern with SeedSequence(master).spawn(n) and stores each integer in the database.
Why: Local generators avoid global state. One master seed reproduces the whole stimulus set. Spawned children are indexed, so adding patterns later doesn't change existing ones.
In plain words: Every stimulus can be regenerated exactly from its parameters and stored seed.

## D013: Matched patterns, fully crossed over valid conditions
Date: 2026-11-11
Decision: Each dot pattern is used across every valid combination of conditions. Pattern is treated as a random effect; bootstrap CIs resample patterns (with all their trials), not individual trials.
Why: Matching removes dot-pattern variability from condition comparisons, like a within-subject design. Shared patterns make trials non-independent, so trial-level resampling would give CIs that are too narrow.
In plain words: Every condition sees the same dot patterns, and the statistics treat patterns like subjects.

## D014: Image order replaces the eye swap
Date: 2026-11-11
Decision: Drop the eye-swap factor. Add an image-order factor: present the right-eye image first with labels kept correct.
Why: With matched seeds, swapping the eyes of a +d pair produces the −d stimulus (up to refill noise), so the eye swap is already covered by the sign manipulation. Image order with correct labels is informative: a model using disparity should not change its answer, while a model relying on position would flip.
In plain words: Swapping eyes is the same as flipping the sign, so I test image position instead.

## D015: Monocular control
Date: 2026-11-11
Decision: Show the model one image of a pair with the same question, scored against the pair's correct answer. Test both the left and the right image. Reuse the stereo stimuli.
Why: Performance should be at chance. Above-chance performance would mean a leaked monocular cue.
In plain words: Each image alone should carry no depth information, and this checks that.

## D016: Valid condition combinations
Date: 2026-11-11
Decision: The trial generator builds valid combinations explicitly rather than a blind product of factors. At d = 0, image order is omitted (both orders give identical stimuli). The monocular control has no format or image-order factor; its own factor is which eye is shown, and at d = 0 left and right are identical.
Approximate size: 68 stereo + 34 monocular trials per pattern, about 100 per pattern.
In plain words: Fully crossed means every combination that produces a distinct stimulus.

## D017: Environment and packaging
Date: 2026-11-11
Decision: uv for environment and dependencies, with uv.lock committed. src/ layout. Notebooks for exploration only, in notebooks/, using the project kernel.
Why: The project is pure Python, so conda's non-Python package management isn't needed. The lockfile makes "one command to reproduce" true. The src/ layout makes tests import the installed package.
In plain words: The range says what the code needs; the lock says what I tested with.

## D018: Square membership by dot center, in edge coordinates
Date: 2026-11-14
Decision: Pixel i spans [i, i+1); a dot at corner r spans [r, r+dot_size), so its center is r + dot_size/2. A dot belongs to a region if its center lies in the region's half-open interval.
Why: Dots and regions use one coordinate system, and all four edges are treated identically.
Alternatives: Pixel-center convention (center = r + (dot_size-1)/2) mixed with edge-based regions, which offsets membership by half a pixel.
In plain words: Dots and the square are measured in the same coordinates.

## D019: Validation limits vs. scientific ranges
Date: 2026-11-14
Decision: make_rds rejects only invalid inputs (non-integers, density outside (0, 1), dot_size >= sq_size, square plus shift not fitting, odd disparity). Preferred dot sizes and densities live in the trial config, justified by expected dots per square and per token.
Why: Hard-coding scientific ranges into the generator would block legitimate future experiments.
In plain words: The generator rejects what's broken; the config decides what's worth testing.

## D020: Score answers as words, summing over varied presentations 
Date: 2026-11-16
Decision: Will score answers as words, summing over same words with different capitalizations and spaces, using multi-token words computed exactly. Record the probability mass over each answer (i.e., "front" and "behind"). Synonyms (e.g., 'in front', 'back', 'rear') only carry about 3% of the probability and so I'm choosing not to count them, since chasing all the possible phrasings would mean more forward passes and for very little return. Answer probability mass is logged per trial to monitor this.  
---

## Open questions

- Density definition: fraction of image area that's black. With random placement, coverage ≈ 1 − exp(−n·a/A); decide whether to set n from target coverage and measure actual coverage in a test.
- Default dot size (currently 4 px).
- Number of patterns: set after timing the pilot.
- Side-by-side format: confirm the combined image stays within the model's pixel budget.
