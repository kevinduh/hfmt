<!-- The developer's request, verbatim (set as the session goal on 2026-11-07). -->

We built a lot of features on the hydra3-translategemma branch. We want to keep all of the
experiments, the run_status script, the proper model saving, etc. We want to remove all of the
modifications made needed to support translategemma model. We can hardcode in turning off thinking
for the qwen models. I want you to put together a plan to port of all of the changes made without
the translategemma requirements to this branch. If there are places to improve the code quality
ONLY IN THE SCOPE OF THE CHANGES WE MADE, do that. Please generate a plan and then begin the port.
Don't commit any code. Put together a porting document that explains the changes made. Be careful
to highlight the diffs
