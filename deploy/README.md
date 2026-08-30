# Deployment

Both targets wrap the same `rel.pipeline.annotate`.

## Blocker: fal serverless access is not yet granted

Checked 2026-08-27 with both a personal and an **admin** fal key. The credential is
definitely valid — an authenticated request to `queue.fal.run` returns 200 where a
deliberately wrong key returns 401 — but every serverless command returns:

```
✘ Insufficient permissions: Please visit
  https://fal.ai/dashboard/serverless-get-started to request access.
```

`fal apps list`, `fal secrets list` and `fal keys list` all fail this way, so
deploying a custom app is not possible on this account today. Deploying custom
Python apps — and later listing one in the Marketplace — requires fal to enable
serverless on the account, which is an approval request with human lead time.

An admin key does not bypass this: serverless is an **account-level entitlement**
fal grants on request, not a key permission.

**Action: a human must request serverless access** at
https://fal.ai/dashboard/serverless-get-started. Nothing on our side unblocks it.

Once granted, deployment is:

```bash
fal secrets set GEMINI_API_KEY=...
fal deploy deploy/fal/app.py::RobotEpisodeLabeler
```

The app has been validated as far as is possible without deploying: the class
resolves, its endpoint is registered, and the request/response schemas validate.
It ships the local `rel` package via `local_python_modules` — an earlier version
inserted a repository path onto `sys.path`, which would have failed on the first
deploy because a fal container has no checkout. `setup()` also resolves an ffmpeg
binary at startup so a decoding problem surfaces then rather than on a request.

Replicate has not been attempted (no credentials yet).

## fal (primary target)

```bash
fal secrets set GEMINI_API_KEY=...
fal deploy deploy/fal/app.py::RobotEpisodeLabeler
```

Billing is declared per second of video via the `x-fal-billable-units` response
header, so a 40-second episode costs 40 units regardless of how many internal
model calls the pipeline made. That keeps pricing stable while the pipeline
changes. Long episodes should be called through fal's queue interface rather than
synchronously.

## Replicate — DEPLOYED

Live (private) at **https://replicate.com/mandarobotics/robot-episode-labeler**.

**Verified end to end, including a hosted prediction.** `scripts/smoke_replicate.py`
uploads a WGO-Bench episode, runs it against the deployed version and prints the
segments:

```
use a gripper to pick the target object and place on the gray plate.  (4.43s, 2 subtasks)
    0.00-  2.00  pass    pick_banana
    2.00-  3.00  pass    place_banana
metrics: {'predict_time': 8.27, 'total_time': 12.93}
```

Gold for that episode is `0.00-1.96 pick` / `1.96-3.29 place`: both events found,
the interior boundary within 0.04 s.

```bash
uv run --with certifi python scripts/smoke_replicate.py
```

Two environment quirks the smoke test works around: some framework Python builds
have no usable root certificate store (certifi's is used when present), and
urllib's default User-Agent is rejected by Replicate's edge with 403.

### Bring-your-own-key

Replicate has no model-level secret store, so `gemini_api_key` is a write-only
`Secret` **input** rather than an environment variable. The caller supplies their
own Gemini key; it is scoped to the call and never returned in the response. This
is the usual pattern for a model that wraps a third-party API.

Note this differs from the intended fal model, where we would hold the key
server-side and bill per second of video. The two distribution channels therefore
have different commercial shapes: Replicate is bring-your-own-key and free to us
per call, fal is our key and billed by the video second.

### Rebuilding

```bash
cog push r8.im/mandarobotics/robot-episode-labeler
```

**2026-08-30: version `5e660f68…` pushed** with the segmentation rework
(decomposition prompt default, `PipelineConfig`, video/state backends). The
push failed twice on the same large layer with transient network errors
(`write: broken pipe`, then `TLS handshake timeout`, both through Docker
Desktop's proxy at 192.168.65.1:3128) and succeeded on the third attempt with
no change; retry before debugging. The hosted smoke test reached the Gemini
call and stopped there with `429 RESOURCE_EXHAUSTED` because the project's
prepaid credits were exhausted at the time — which verifies the image boots,
decodes and builds sheets, but not a full prediction; re-run
`scripts/smoke_replicate.py` after topping up. `95b8c8d3…`, one minute
earlier, is the identical image pushed twice by a retry loop and can be
ignored.

`cog.yaml` sits at the repository root so `src/rel` ships in the image, and
`.dockerignore` keeps `data/` (1.3 GB of benchmark video) out of the build context.
cog 0.22 renamed the entry point (`predict:` → `run:`, `Predictor.predict()` →
`Predictor.run()`) and its static analyser rejects an aliased `cog.Path`, so the
import is unaliased.

Two account quirks worth recording:

- **`cog login` needs a CLI auth token, not a Replicate API token.** The API token
  is rejected outright. Fetch one from https://replicate.com/auth/token.
- **`POST /v1/models` returned HTTP 500** for this token on every attempt, including
  a minimal payload under a throwaway name, while reads on the same token
  succeeded. The model had to be created through the web UI.

## Hugging Face Space — prepared, not yet pushed

`deploy/hf_space/` is a Gradio front end over the same `annotate()`:
bring-your-own Gemini key (passed per call, never through the shared process
environment), episodes capped at 5 min / 200 MB for the free CPU tier, the two
public DROID clips as examples, and `ABOUT.md` (accuracy, failure modes, data
handling) rendered under the form. `README.md` is the Space card.

```bash
uv run python scripts/publish_hf_space.py            # dry run: stages and lists files
uv run python scripts/publish_hf_space.py --push     # creates/updates mandarobotics/robot-episode-labeler
```

The script copies `src/rel` and the example clips into a staging directory, so
the Space is self-contained and does not depend on the GitHub repository being
public. It authenticates with `HF_KEY` from `.env` **explicitly** and refuses
any target outside `mandarobotics/` or any token that is not a member of that
org — a cached `huggingface-cli login` for another account was found on this
machine and must never be used by accident. Verified locally from the staged
directory: the UI serves, a missing key and an exhausted-credit key both
produce specific errors rather than tracebacks.

Because the app is copied at publish time, changes to `src/rel` reach the Space
only on the next `--push`.

## Before either goes public

- [ ] Set `GEMINI_API_KEY` as a platform secret, never baked into the image.
- [ ] Decide and publish a retention policy for uploaded video. Customers send
      unreleased hardware footage; "we do not retain video after annotation"
      needs to be true and stated.
- [ ] Rate-limit and cap episode duration; `MAX_BYTES` in `video/source.py` is
      the only guard today.
- [ ] Publish measured accuracy alongside the endpoint. See `results/`.
