"""
Gemma4BatchCaptioner - ComfyUI Node (v1.2)
==========================================
Batch image caption writer powered by local Gemma 4 via llama-server.

Feed it a folder of images — the node calls Gemma 4 vision on every image
and writes a .txt file next to each one (same base name) containing the
generated caption at your requested word count.

Workflow:
  1. Set FOLDER PATH to the directory that contains your images.
  2. Pick your GGUF model from the dropdown (scanned from C:\\models\\).
  3. Choose WORD COUNT, CAPTION STYLE, and optional PREFIX.
  4. Click Queue Prompt — llama-server is auto-started, every image is
     captioned, a .txt is written for each one, then llama-server is
     automatically killed and VRAM/RAM is fully cleared.

Supports: jpg / jpeg / png / webp / bmp / tiff / gif
Backend:  llama-server (llama.cpp) — auto-started from C:\\llama\\ or PATH.
          Gemma 4 multimodal GGUF + mmproj must both be in C:\\models\\.

Part of the LoRa-Daddy toolkit.
"""

import base64
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (mirrors gemma4_prompt_gen.py)
# ══════════════════════════════════════════════════════════════════════════════

LLAMA_INSTALL_DIR = r"C:\llama"
MODELS_DIR        = r"C:\models"
LLAMA_RELEASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b8664/"
    "llama-b8664-bin-win-cuda-cu12.4-x64.zip"
)
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


# ══════════════════════════════════════════════════════════════════════════════
#  GGUF SCANNER  (same logic as gemma4_prompt_gen.py)
# ══════════════════════════════════════════════════════════════════════════════

def _scan_models_folder() -> list:
    """Return list of .gguf filenames in C:\\models\\ (excluding mmproj), or a placeholder."""
    if not os.path.isdir(MODELS_DIR):
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
        except Exception:
            pass
    try:
        ggufs = sorted([
            f for f in os.listdir(MODELS_DIR)
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower()
        ])
        return ggufs if ggufs else ["No GGUFs found in C:\\models\\"]
    except Exception:
        return ["No GGUFs found in C:\\models\\"]


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTION STYLE PRESETS
# ══════════════════════════════════════════════════════════════════════════════

CAPTION_STYLES = {
    "Descriptive — natural prose": (
        "You are a precise image captioning assistant. "
        "Describe the image in clear, flowing natural-language prose. "
        "Cover: subject(s), action or pose, setting, lighting, colour palette, mood. "
        "Do NOT begin with 'This image shows' or 'The image depicts'. "
        "Start directly with the subject. "
        "Output ONLY the caption text — no preamble, no labels, no markdown."
    ),
    "Alt-text — accessibility": (
        "You are an accessibility alt-text writer. "
        "Write a concise alt-text description of the image for screen-reader users. "
        "Include: what is depicted, who/what is in the foreground, relevant context. "
        "Be factual and neutral. Avoid subjective adjectives. "
        "Output ONLY the alt-text — no preamble, no labels."
    ),
    "Training tags — booru style": (
        "You are a dataset tagging assistant for image model training. "
        "Output a comma-separated list of descriptive booru-style tags for this image. "
        "Cover in order: quality rating, subject description, clothing, pose/action, "
        "setting, lighting, art style, colour palette, camera angle. "
        "Use underscores for multi-word tags. "
        "Output ONLY the tag string — no sentences, no labels, no markdown."
    ),
    "Training tags — natural sentence": (
        "You are a dataset captioning assistant for image model training. "
        "Write a single dense natural-language sentence that describes this image "
        "in a way optimised for training a text-to-image diffusion model. "
        "Prioritise: subject identity, appearance, clothing, action, environment, lighting. "
        "Be specific and concrete. No filler words. "
        "Output ONLY the caption sentence — no labels, no markdown."
    ),
    "Cinematic — prompt style": (
        "You are a cinematic image analyst writing prompts for image generation. "
        "Describe the image as a rich cinematic generation prompt: "
        "lens framing, subject, lighting setup, colour grade, mood, depth of field, texture. "
        "Use comma-separated fragments in the style of a Flux or SDXL prompt. "
        "Output ONLY the prompt text — no labels, no markdown, no preamble."
    ),
    "NSFW — Explicit & Uncensored": (
        "You are an uncensored explicit image captioning assistant. "
        "Describe nudity, sexual content, and anatomy in direct, raw, and highly explicit language. "
        "Use clear anatomical terms without any censorship or euphemisms: pussy, cock, tits, boobs, cleavage, nipples, asshole, clit, labia, erection, cum, wet pussy, dripping, spread legs, penetrated, etc. "
        "Be visually detailed and unfiltered. Describe exactly what is visible — body parts, arousal, fluids, sexual acts, and poses. "
        "Do not be vague, shy, or polite. "
        "Start directly with the subject and describe the scene explicitly. "
        "Output ONLY the caption text — no preamble, no labels, no markdown."
    ),
    "Closeup — Expression": (
        "You are a LoRA training caption writer for facial expression images."
        "Output a single short caption in this exact format: [subject], closeup portrait, [primary emotion] expression, [one lighting descriptor]"
        "Primary emotion must be one specific word chosen from: happy, joyful, laughing, smiling, sad, tearful, crying, angry, furious, disgusted, fearful, surprised, shocked, contemptuous, confused, anxious, pained, concentrated, neutral, blank, tense, embarrassed, longing, proud, smiling with teeth visible"
        "Lighting descriptor must be one of: soft light, harsh light, natural light, studio light, side light, warm light, dim light"
        "Do NOT add any other words, sentences, descriptions, or commentary"
        "Do NOT describe the background, clothing, hair, or camera"
        "Do NOT use more than 15 words total after the subject"
        "Output ONLY the caption — no preamble, no labels, no punctuation beyond commas"
    ),
    "Custom — use instruction below": (
        "You are a professional image captioning assistant. "
        "Follow the user instruction exactly. "
        "Output ONLY the caption — no preamble, no labels, no markdown."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION PRESETS
#  Each entry controls top_p, top_k, and the max_tokens formula.
#  temperature is kept as a separate user input.
#  max_tokens = int(word_count * token_mult) + token_buf
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_PRESETS = {
    #  name                         top_p   top_k   token_mult  token_buf
    "⚖️  Balanced — recommended":   (0.88,  35,     1.40,       60),
    "🎯  Precise — tight & factual": (0.82,  25,     1.30,       40),
    "📝  Descriptive — detail rich": (0.88,  38,     1.50,       70),
    "🎬  Creative — cinematic":      (0.93,  50,     1.55,       80),
    "🏷️  Tags — booru optimised":    (0.80,  20,     1.70,       110),
    "🔞  Explicit — uncensored":     (0.90,  40,     1.50,       80),
}

# Tooltip shown next to the dropdown explaining each preset
_PRESET_TOOLTIP = (
    "⚖️  Balanced         — safe default for any style. "
    "🎯  Precise          — tightest output, best for alt-text & training-tags. "
    "📝  Descriptive      — richer vocabulary, best for natural prose & LoRA captions. "
    "🎬  Creative         — widest sampling, best for cinematic prompt style. "
    "🏷️  Tags/Booru       — tuned for comma-tag lists (gives extra token budget). "
    "🔞  Explicit         — matches the NSFW caption style."
)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE ENCODE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _encode_image_to_b64(image_path: str, max_side: int = 768):
    """Load any supported image, resize, and return JPEG base64. None on failure."""
    try:
        from PIL import Image as PILImage
        import io

        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        print(f"[Gemma4BatchCaptioner] Image encode failed for {image_path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTION CLEANER
# ══════════════════════════════════════════════════════════════════════════════

def _clean_caption(text: str) -> str:
    """Strip thinking tokens, markdown fences, and common LLM preamble."""
    text = re.sub(r'<\|channel\|>thought\n.*?<\|/channel\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|channel>thought\n.*?<channel\|>',      '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>',                       '', text, flags=re.DOTALL)
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)

    junk_prefixes = [
        r"^here'?s?\s+(a\s+)?(caption|description|alt.?text|the caption|an alt)[:\-\u2013\u2014]?\s*",
        r"^(sure|of course|certainly)[,!.]?\s*",
        r"^(caption|alt.?text|description|tags?|prompt)[:\-\u2013\u2014]\s*",
        r"^the image (shows|depicts|features|contains)\s*",
        r"^this image (shows|depicts|features|contains)\s*",
        r"^in this image[,\s]",
        r"^i(?:'ve| have) (generated|written|created)[^.]*\.\s*",
    ]
    for pattern in junk_prefixes:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.strip()
    if len(text) > 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  NODE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Gemma4BatchCaptioner:
    """
    Batch caption writer.
    Scans a folder of images, generates a caption per image via Gemma 4 vision,
    prepends the optional prefix, writes a matching .txt, then kills llama-server
    and clears VRAM/RAM automatically when the batch finishes or errors out.
    """

    CATEGORY = "LoRa-Daddy/Gemma4"

    _llama_process = None   # shared process handle (same pattern as Gemma4PromptGen)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── FOLDER & OUTPUT ───────────────────────────────────────────
                "folder_path": ("STRING", {
                    "default": r"C:\datasets\my_images",
                    "multiline": False,
                    "tooltip": "Full path to the folder containing images to caption.",
                }),
                "caption_prefix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Text prepended to every generated caption, separated by a comma and space. "
                        "Example: 'ohwx woman' → final caption: 'ohwx woman, a woman standing in a park...'. "
                        "Leave blank for no prefix."
                    ),
                }),
                "word_count": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 500,
                    "step": 5,
                    "tooltip": "Target word count for the generated caption body (prefix words not counted).",
                }),
                "caption_style": (list(CAPTION_STYLES.keys()), {
                    "default": "Descriptive — natural prose",
                    "tooltip": "Pre-built captioning style. Choose 'Custom' to write your own directive below.",
                }),
                "custom_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Only active when caption_style is 'Custom — use instruction below'. "
                        "Example: 'Describe only the clothing and accessories worn by the subject.'"
                    ),
                }),
                "overwrite_existing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Regenerate captions even when a .txt already exists. Off = safe resume.",
                }),
                "image_max_side": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1536,
                    "step": 128,
                    "tooltip": "Resize images to this max dimension before sending. 768 is fast and accurate.",
                }),
                "stop_on_error": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Halt the batch on the first API error. Off = skip failed images and continue.",
                }),
                # ── GENERATION PRESET ─────────────────────────────────────────
                "generation_preset": (list(GENERATION_PRESETS.keys()), {
                    "default": "⚖️  Balanced — recommended",
                    "tooltip": _PRESET_TOOLTIP,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.35,
                    "min":     0.01,
                    "max":     1.50,
                    "step":    0.05,
                    "tooltip": (
                        "Sampling temperature — the only parameter you need to tune manually. "
                        "Lower = more consistent & factual (0.20–0.35 for captions). "
                        "Higher = more varied but risks padding (0.50+ for creative styles). "
                        "All other sampling values are handled by the preset above."
                    ),
                }),
                # ── BACKEND  (same layout as gemma4_prompt_gen) ───────────────
                "\U0001f5a5\ufe0f llama_server_url": ("STRING", {
                    "default": "http://127.0.0.1:8080",
                    "tooltip": "llama-server base URL. Default: http://127.0.0.1:8080",
                }),
                "\U0001f9e0 gguf_model": (
                    _scan_models_folder(),
                    {
                        "default": _scan_models_folder()[0],
                        "tooltip": "GGUF model from C:\\models\\. Add files there and restart ComfyUI to refresh.",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("summary", "folder_path")
    FUNCTION      = "caption_folder"
    OUTPUT_NODE   = True

    # ── Execution ─────────────────────────────────────────────────────────────

    def caption_folder(self, **kwargs):

        # Resolve emoji-prefixed keys (same _kw pattern as gemma4_prompt_gen)
        def _kw(primary, *aliases, default=None):
            for key in (primary, *aliases):
                if key in kwargs:
                    return kwargs[key]
            return default

        folder_path        = _kw("folder_path",        default=r"C:\datasets\my_images")
        caption_prefix     = _kw("caption_prefix",     default="")
        word_count         = _kw("word_count",         default=60)
        caption_style      = _kw("caption_style",      default="Descriptive — natural prose")
        custom_instruction = _kw("custom_instruction", default="")
        overwrite_existing = _kw("overwrite_existing", default=False)
        image_max_side     = _kw("image_max_side",     default=768)
        stop_on_error      = _kw("stop_on_error",      default=False)
        generation_preset  = _kw("generation_preset",  default="⚖️  Balanced — recommended")
        temperature        = float(_kw("temperature",  default=0.35))
        server_url         = _kw(
            "\U0001f5a5\ufe0f llama_server_url", "llama_server_url",
            default="http://127.0.0.1:8080"
        )
        gguf_model         = _kw(
            "\U0001f9e0 gguf_model", "gguf_model",
            default=""
        )

        # ── Resolve preset → sampling params ─────────────────────────────────
        top_p, top_k, tok_mult, tok_buf = GENERATION_PRESETS.get(
            generation_preset,
            GENERATION_PRESETS["⚖️  Balanced — recommended"],
        )

        server_url     = (server_url or "http://127.0.0.1:8080").strip().rstrip("/")
        folder_path    = folder_path.strip().rstrip("\\/")
        caption_prefix = caption_prefix.strip()

        # ── Resolve model path ────────────────────────────────────────────────
        if gguf_model and gguf_model != "No GGUFs found in C:\\models\\":
            model_path = os.path.join(MODELS_DIR, gguf_model)
        else:
            found = (
                [f for f in os.listdir(MODELS_DIR)
                 if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
                if os.path.isdir(MODELS_DIR) else []
            )
            if not found:
                msg = "❌ No GGUF files found in C:\\models\\. Add a GGUF and restart ComfyUI."
                return (msg, folder_path)
            model_path = os.path.join(MODELS_DIR, found[0])

        # ── Validate folder ───────────────────────────────────────────────────
        if not os.path.isdir(folder_path):
            msg = f"❌ Folder not found: {folder_path}"
            print(f"[Gemma4BatchCaptioner] {msg}")
            return (msg, folder_path)

        # ── Collect images ────────────────────────────────────────────────────
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            msg = f"⚠️ No supported images found in: {folder_path}"
            print(f"[Gemma4BatchCaptioner] {msg}")
            return (msg, folder_path)

        print(f"[Gemma4BatchCaptioner] Found {len(image_files)} image(s) in {folder_path}")
        if caption_prefix:
            print(f"[Gemma4BatchCaptioner] Prefix: \"{caption_prefix}\"")

        # ── Flush VRAM before booting llama-server ────────────────────────────
        self._flush_vram()

        # ── Auto-start llama-server ───────────────────────────────────────────
        llama_exe = self._find_or_install_llama()
        if llama_exe.startswith("❌"):
            return (llama_exe, folder_path)

        boot_status = self._ensure_llama_running(server_url, llama_exe, model_path)
        print(f"[Gemma4BatchCaptioner] {boot_status}")
        if boot_status.startswith("❌"):
            return (boot_status, folder_path)

        # ── Build system prompt ───────────────────────────────────────────────
        system_prompt = CAPTION_STYLES.get(
            caption_style, CAPTION_STYLES["Descriptive — natural prose"]
        )
        if caption_style == "Custom — use instruction below" and custom_instruction.strip():
            system_prompt = (
                "You are a professional image captioning assistant. "
                f"Task: {custom_instruction.strip()} "
                "Output ONLY the caption — no preamble, no labels, no markdown."
            )

        # ── Process each image ────────────────────────────────────────────────
        processed = 0
        skipped   = 0
        errors    = 0
        error_log = []

        try:
            for idx, filename in enumerate(image_files, start=1):
                image_path = os.path.join(folder_path, filename)
                base_name  = os.path.splitext(filename)[0]
                txt_path   = os.path.join(folder_path, base_name + ".txt")

                if not overwrite_existing and os.path.isfile(txt_path):
                    print(f"[Gemma4BatchCaptioner] [{idx}/{len(image_files)}] SKIP (exists): {filename}")
                    skipped += 1
                    continue

                print(f"[Gemma4BatchCaptioner] [{idx}/{len(image_files)}] Captioning: {filename}")

                b64 = _encode_image_to_b64(image_path, max_side=image_max_side)
                if b64 is None:
                    errors += 1
                    error_log.append(f"Encode failed: {filename}")
                    if stop_on_error:
                        break
                    continue

                user_text = (
                    f"Write a caption for this image.\n"
                    f"Target length: approximately {word_count} words.\n"
                    f"Do NOT exceed {int(word_count * 1.25)} words.\n"
                    f"Output ONLY the caption text — nothing else."
                )

                caption_raw = self._call_llama(
                    server_url    = server_url,
                    system_prompt = system_prompt,
                    user_text     = user_text,
                    image_b64     = b64,
                    word_target   = word_count,
                    temperature   = temperature,
                    top_p         = top_p,
                    top_k         = top_k,
                    tok_mult      = tok_mult,
                    tok_buf       = tok_buf,
                )

                if caption_raw.startswith("❌") or caption_raw.startswith("⚠️"):
                    print(f"[Gemma4BatchCaptioner] ⚠️ {filename}: {caption_raw}")
                    errors += 1
                    error_log.append(f"{filename}: {caption_raw}")
                    if stop_on_error:
                        break
                    continue

                caption_body = _clean_caption(caption_raw)

                if not caption_body:
                    errors += 1
                    error_log.append(f"Empty result: {filename}")
                    if stop_on_error:
                        break
                    continue

                # ── Apply prefix ──────────────────────────────────────────────
                if caption_prefix:
                    # Smart join: if body already starts with prefix (case-insensitive), don't double it
                    if caption_body.lower().startswith(caption_prefix.lower()):
                        final_caption = caption_body
                    else:
                        final_caption = f"{caption_prefix}, {caption_body}"
                else:
                    final_caption = caption_body

                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(final_caption)
                    word_total = len(final_caption.split())
                    print(f"[Gemma4BatchCaptioner]   ✅ {base_name}.txt  ({word_total} words)")
                    processed += 1
                except Exception as e:
                    errors += 1
                    error_log.append(f"Write failed {txt_path}: {e}")
                    if stop_on_error:
                        break

        finally:
            # ── Always kill llama-server and free memory when done ────────────
            # Runs whether the batch completed normally, hit stop_on_error,
            # was interrupted, or raised an unexpected exception.
            print("[Gemma4BatchCaptioner] Batch finished — shutting down llama-server...")
            self._kill_llama_server()
            self._flush_vram()

        # ── Summary ───────────────────────────────────────────────────────────
        lines = [
            "✅ Batch captioning complete.",
            f"   Folder      : {folder_path}",
            f"   Model       : {os.path.basename(model_path)}",
            f"   Style       : {caption_style}",
            f"   Preset      : {generation_preset}",
            f"   Params      : temp={temperature}  top_p={top_p}  top_k={top_k}"
            f"  max_tokens={int(word_count * tok_mult) + tok_buf}",
            f"   Prefix      : {caption_prefix if caption_prefix else '(none)'}",
            f"   Words       : ~{word_count}",
            f"   Total       : {len(image_files)} image(s)",
            f"   Written     : {processed}",
            f"   Skipped     : {skipped}  (txt already existed)",
            f"   Errors      : {errors}",
            "   Server      : stopped — VRAM cleared ✓",
        ]
        if error_log:
            lines.append("\nErrors:")
            for e in error_log:
                lines.append(f"   • {e}")

        summary = "\n".join(lines)
        print(f"\n[Gemma4BatchCaptioner]\n{summary}\n")
        return (summary, folder_path)

    # ── llama-server API call ─────────────────────────────────────────────────

    def _call_llama(
        self,
        server_url,
        system_prompt,
        user_text,
        image_b64,
        word_target,
        temperature = 0.35,
        top_p       = 0.88,
        top_k       = 35,
        tok_mult    = 1.40,
        tok_buf     = 60,
    ):
        endpoint = f"{server_url}/v1/chat/completions"

        user_content = [
            {
                "type":      "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
            {"type": "text", "text": user_text},
        ]

        payload = {
            "model":       "gemma4",
            "messages":    [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "temperature": float(temperature),
            "top_p":       float(top_p),
            "top_k":       int(top_k),
            "max_tokens":  int(word_target * tok_mult) + tok_buf,
            "stream":      False,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                endpoint,
                data    = data,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            content = result["choices"][0]["message"]["content"]
            content = re.sub(r'<\|channel\|>thought\n.*?<\|/channel\|>', '', content, flags=re.DOTALL)
            content = re.sub(r'<\|channel>thought\n.*?<channel\|>',      '', content, flags=re.DOTALL)
            content = re.sub(r'<think>.*?</think>',                       '', content, flags=re.DOTALL)
            content = content.strip()

            if not content:
                return "⚠️ Model returned empty response. Retry."
            return content

        except urllib.error.URLError as e:
            return f"❌ llama-server connection failed: {e}"
        except KeyError:
            return "❌ Unexpected API response format."
        except Exception as e:
            return f"❌ Error calling llama-server: {e}"

    # ── Memory helpers ────────────────────────────────────────────────────────

    def _flush_vram(self):
        """Unload ComfyUI models and flush CUDA cache."""
        try:
            import comfy.model_management as _mm
            _mm.unload_all_models()
            _mm.soft_empty_cache()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def _kill_llama_server(self):
        """
        Kill all llama-server processes by name and by stored PID,
        then flush VRAM/RAM. Mirrors gemma4_prompt_gen._kill_llama_server exactly.
        """
        for proc_name in ["llama-server.exe", "llama-server", "llama-cli.exe"]:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=10
                )
            except Exception:
                pass

        # Also kill by stored PID in case taskkill missed it
        if Gemma4BatchCaptioner._llama_process is not None:
            try:
                Gemma4BatchCaptioner._llama_process.kill()
            except Exception:
                pass
            Gemma4BatchCaptioner._llama_process = None

        print("[Gemma4BatchCaptioner] llama-server killed — VRAM freed.")

    # ── llama-server lifecycle  (mirrors gemma4_prompt_gen.py exactly) ────────

    def _check_health(self, server_url: str = "http://127.0.0.1:8080") -> bool:
        try:
            req = urllib.request.Request(f"{server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _ensure_llama_running(self, server_url: str, llama_exe: str, model_path: str) -> str:
        """Boot llama-server if not running. Auto-detects and requires mmproj for vision."""
        if self._check_health(server_url):
            return "✅ llama-server already running"

        if not os.path.isfile(llama_exe):
            return f"❌ llama-server not found at: {llama_exe}"
        if not os.path.isfile(model_path):
            return f"❌ Model GGUF not found at: {model_path}"

        # Vision is mandatory for captioning — require mmproj
        mmproj_path = None
        for f in os.listdir(os.path.dirname(model_path)):
            if "mmproj" in f.lower() and f.lower().endswith(".gguf"):
                mmproj_path = os.path.join(os.path.dirname(model_path), f)
                print(f"[Gemma4BatchCaptioner] mmproj: {mmproj_path}")
                break

        if not mmproj_path:
            return (
                "❌ No mmproj GGUF found in C:\\models\\. "
                "Vision captioning requires an mmproj file alongside the main GGUF."
            )

        cmd = [
            llama_exe,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "-ngl", "99",
            "--ctx-size", "8192",
            "--flash-attn", "on",
            "--reasoning-budget", "0",
        ]

        try:
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS         = 0x00000008
            Gemma4BatchCaptioner._llama_process = subprocess.Popen(
                cmd,
                stdout        = subprocess.DEVNULL,
                stderr        = subprocess.DEVNULL,
                creationflags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
            )
        except Exception as e:
            return f"❌ Failed to start llama-server: {e}"

        print("[Gemma4BatchCaptioner] llama-server starting with vision...")
        for waited in range(0, 120, 2):
            time.sleep(2)
            if self._check_health(server_url):
                return f"✅ llama-server started ({waited + 2}s) — vision enabled"

        return "❌ llama-server health check timed out after 120s"

    def _find_or_install_llama(self) -> str:
        """Find llama-server.exe in standard locations, or auto-download to C:\\llama\\."""
        import zipfile

        candidate = os.path.join(LLAMA_INSTALL_DIR, "llama-server.exe")
        if os.path.isfile(candidate):
            return candidate

        try:
            result = subprocess.run(
                ["where", "llama-server"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
            )
            if result.returncode == 0:
                found = result.stdout.strip().split("\n")[0].strip()
                if found and os.path.isfile(found):
                    print(f"[Gemma4BatchCaptioner] Found llama-server in PATH: {found}")
                    return found
        except Exception:
            pass

        for p in [
            r"C:\llama\llama-server.exe",
            r"C:\Program Files\llama.cpp\llama-server.exe",
            os.path.expanduser(r"~\llama\llama-server.exe"),
        ]:
            if os.path.isfile(p):
                return p

        # Auto-download
        print(f"[Gemma4BatchCaptioner] llama-server not found — auto-installing to {LLAMA_INSTALL_DIR}...")
        try:
            os.makedirs(LLAMA_INSTALL_DIR, exist_ok=True)
            zip_path = os.path.join(LLAMA_INSTALL_DIR, "llama_install.zip")
            urllib.request.urlretrieve(LLAMA_RELEASE_URL, zip_path)
            print("[Gemma4BatchCaptioner] Download complete. Extracting...")

            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    fname = os.path.basename(member)
                    if not fname:
                        continue
                    with zf.open(member) as src, \
                         open(os.path.join(LLAMA_INSTALL_DIR, fname), "wb") as dst:
                        dst.write(src.read())

            os.remove(zip_path)

            if os.path.isfile(candidate):
                print(f"[Gemma4BatchCaptioner] ✅ llama-server installed at {candidate}")
                return candidate
            return f"❌ Extraction done but llama-server.exe not found in {LLAMA_INSTALL_DIR}"

        except Exception as e:
            return f"❌ Auto-install failed: {e}. Download manually from: {LLAMA_RELEASE_URL}"


# ── ComfyUI Registration ───────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Comfy_Gemma4BatchCaptioner": Gemma4BatchCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy_Gemma4BatchCaptioner": "🖼️ Gemma4 Batch Captioner Engine",
}
