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

## Replicate

```bash
cd deploy/replicate && cog push r8.im/<org>/robot-episode-labeler
```

Replicate takes an uploaded file rather than a URL, which is the friendlier way to
try the model from its interactive page.

## Before either goes public

- [ ] Set `GEMINI_API_KEY` as a platform secret, never baked into the image.
- [ ] Decide and publish a retention policy for uploaded video. Customers send
      unreleased hardware footage; "we do not retain video after annotation"
      needs to be true and stated.
- [ ] Rate-limit and cap episode duration; `MAX_BYTES` in `video/source.py` is
      the only guard today.
- [ ] Publish measured accuracy alongside the endpoint. See `results/`.
