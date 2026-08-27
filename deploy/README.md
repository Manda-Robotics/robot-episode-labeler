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

## Before either goes public

- [ ] Set `GEMINI_API_KEY` as a platform secret, never baked into the image.
- [ ] Decide and publish a retention policy for uploaded video. Customers send
      unreleased hardware footage; "we do not retain video after annotation"
      needs to be true and stated.
- [ ] Rate-limit and cap episode duration; `MAX_BYTES` in `video/source.py` is
      the only guard today.
- [ ] Publish measured accuracy alongside the endpoint. See `results/`.
