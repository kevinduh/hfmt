<!-- Captured verbatim from the developer's chat request on 2026-10-08.
     request.md is developer-authored; edit/trim this to taste. -->

Currently we're saving model outputs to files like `outputs/mmtc_fr-en_sft1/0_2026-10-07_14-01-07/`
(with a sibling `..._b/` for the adapter). This does not match our W&B path/name, so it's hard to
align the two. Can we update file paths so they're aligned? We should use the W&B names. Does a
nested directory structure make sense here?

Follow-on constraints:

- I want the dirs to be reasonable and not pains in the butt that need to be escaped — perform some
  kind of string escape/sanitization so the names are shell-safe.
