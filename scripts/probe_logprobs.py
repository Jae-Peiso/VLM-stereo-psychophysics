"""Probe: one stereo pair through Qwen2.5-VL, reporting what the model receives and its next-token log-probabilities.
We're just exploring, this is not backend code."""
import mlx.core as mx 
import numpy as np
from PIL import Image
from mlx_vlm import load, apply_chat_template, prepare_inputs 

from vlm_stereo.stimuli import make_rds

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
PROMPT_1 = ("The first image is the left eye's view and the second image is the right eye's view of the same scene."
        " When the two are combined in stereo, is the central square in front of or behind the background?"
        " Answer with one word: front or behind.")
PROMPT_2 = ("The first image is the left eye's view and the second image is the right eye's view of the same scene."
        " When the two are combined in stereo, is the central square behind or in front of the background?"
        " Answer with one word: behind or front.")
PROMPT = PROMPT_2 

# 1. Load model and processor (weights come from local cache). 
model, processor = load(MODEL_ID)
config = model.config
tokenizer = getattr(processor, "tokenizer", processor)

# 2. Build stereo pair and convert to RGB images 
L, R = make_rds(seed = 0, disparity = 0)
images = [Image.fromarray(L).convert("RGB"), Image.fromarray(R).convert("RGB")]

# 3. Chat template with placeholders for two images. 
prompt = apply_chat_template(processor, config, PROMPT, num_images = 2)
print("==== formatted prompt ====")

# 4. Turn text + images into model inputs.
inputs = prepare_inputs(processor, images = images, prompts=prompt, 
                        image_token_index=getattr(config, "image_token_index", None))
print("\n=== input keys ===", list(inputs.keys()))
print("image_grid_thw:", inputs.get("image_grid_thw"))

# 5. One forward pass; logits for the next token after the prompt.
input_ids = inputs["input_ids"]
pixel_values = inputs["pixel_values"]
mask = inputs.get("attention_mask")
extra = {k: v for k, v in inputs.items()
         if k not in ("input_ids", "pixel_values", "attention_mask")}
out = model(input_ids, pixel_values, mask = mask, **extra)
logits = getattr(out, "logits", out)[0, -1, :]
logprobs = logits - mx.logsumexp(logits)                        # log-softmax
mx.eval(logprobs)
lp = np.array(logprobs.astype(mx.float32))

# 6. What does the model most want to say?
print("\n=== top 10 next tokens ===")
for i in np.argsort(lp)[::-1][:10]:
    print(f"{lp[i]:8.3f} {tokenizer.decode([int(i)])!r}")

# 7. How do the answer words tokenize, and what are their log-probabilities? 
print("\n=== candidate answers ===")
for word in ["front", "Front", " front", " Front",
             "behind", "Behind", " behind", " Behind"]:
    ids = tokenizer.encode(word, add_special_tokens = False)
    note = f"first-token logprob {lp[ids[0]]:8.3f}" if ids else ""
    print(f"{word!r:10} -> {ids}  {note}")

# 8. Score the two answer WORDS: sum over capitalizations, and compute 
# two-token "beh"-"ind" exactly by appending its first token ("beh").
BEH, IND = tokenizer.encode("behind", add_special_tokens=False)         # [29898, 484]
ids2 = mx.concatenate([input_ids, mx.array([[BEH]])], axis = 1)
mask2 = None if mask is None else mx.concatenate(
    [mask, mx.ones((1,1), dtype=mask.dtype)], axis = 1)
out2 = model(ids2, pixel_values, mask=mask2, **extra)
lg = getattr(out2, "logits", out2)[0]
lp_first = lg[-2] - mx.logsumexp(lg[-2])                                # answer position, same as step # 5
lp_next = lg[-1] - mx.logsumexp(lg[-1])                                 # position after "beh"

# Check: the answer-position distribution must match step #5's.
print("\n=== check ===")
print("max difference from step 5:", np.abs(lp_first - lp).max())

FRONT = tokenizer.encode("front", add_special_tokens = False)[0]
FRONT_CAP = tokenizer.encode("Front", add_special_tokens = False)[0]
BEHIND_CAP = tokenizer.encode("Behind", add_special_tokens = False)[0]

p_front = np.exp(lp[FRONT]) + np.exp(lp[FRONT_CAP])
p_behind = np.exp(lp_first[BEH] + lp_next[IND]) + np.exp(lp[BEHIND_CAP])

print("\n=== answer scores ===")
print(f"P(front answer)         = {p_front:.4f}")
print(f"P(behind answer)        = {p_behind:.4f}")
print(f"answer mass             = {p_front + p_behind:.4f}")
print(f"decision variable log P(front) - log P(behind) = "
      f"{np.log(p_front) - np.log(p_behind):.3f}")
  
