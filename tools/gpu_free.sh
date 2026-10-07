#!/usr/bin/env bash
#
# gpu_free.sh -- how many GPUs of each type are free right now, so you can pick the
# least-contended type before launching a run (e.g. choose a100 vs l40s for egs/run.sh).
#
# Usage (run on a Slurm login node):
#   bash tools/gpu_free.sh                 # summary for a100 and l40s (our 4gpu_tier bf16 set)
#   bash tools/gpu_free.sh a100 l40s v100  # summary for the GPU types you name
#   bash tools/gpu_free.sh -v              # also list each node that has a free GPU
#
# "FREE" = configured GPUs minus allocated GPUs, counted only on nodes in a usable state
# (DOWN/DRAIN/MAINT/RESV/etc are treated as 0 free). "NODES+" = how many nodes have >=1 free
# GPU of that type -- what matters for a 1-GPU job.
#
# NOTE: this is raw CLUSTER availability. Your 4gpu_tier still caps you at gres/gpu=4 total,
# so you can occupy at most 4 GPUs regardless of what is free here.
set -euo pipefail

command -v sinfo >/dev/null 2>&1 || {
  echo "gpu_free.sh: 'sinfo' not found -- run this on a Slurm login node." >&2
  exit 1
}
# Data source is `sinfo -N` (one row per node) with the GresUsed field: this cluster's
# `scontrol show node --oneliner` omits GresUsed (allocation only shows up in AllocTRES),
# whereas sinfo reports GRES_USED directly, so netting free = configured - used is reliable.

verbose=0; emit=0
while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -v)     verbose=1 ;;
    --emit) emit=1 ;;        # machine mode: print "<type> <free>" lines only, nothing else
    *) echo "gpu_free.sh: unknown flag '$1'" >&2; exit 2 ;;
  esac
  shift
done
if [[ $# -gt 0 ]]; then types=("$@"); else types=(a100 l40s); fi

sinfo -N -h -p "${PART:-gpu}" -O "NodeList:30,StateCompact:16,Gres:48,GresUsed:220" \
  | awk -v TYPES="${types[*]}" -v verbose="$verbose" -v emit="$emit" '
function parse(s, arr,   nt,toks,k,tok,np,p,cnt){
  split("", arr)                                   # clear arr (portable)
  if (s=="" || s=="(null)") return
  nt=split(s, toks, ",")
  for (k=1;k<=nt;k++){
    tok=toks[k]                                     # e.g. gpu:a100:2(IDX:0-1) or gpu:a100:4
    if (tok ~ /^gpu:/){
      np=split(tok, p, ":")
      if (np>=3){ cnt=p[3]; sub(/\(.*/,"",cnt); arr[p[2]]+=cnt+0 }
    }
  }
}
function pr(t){
  printf "%-8s %6d %6d %7d %7d\n", t, tot_free[t]+0, tot_used[t]+0, tot_cfg[t]+0, nfree[t]+0
}
BEGIN{
  n=split(TYPES, want, " ")
  for (i=1;i<=n;i++) wantset[want[i]]=1
}
{
  node=$1; state=tolower($2); gres=$3; used=$4
  if (node in seen) next                            # a node may list once per partition
  seen[node]=1
  # Exclude unschedulable nodes: compact state words + power/no-respond suffix symbols.
  ok = (state !~ /down|drain|drng|drn|resv|maint|boot|plnd|fail|inval|pow|comp/) && (state !~ /[*~#@%!]/)
  parse(gres, cfg)
  parse(used, usd)
  for (t in cfg){
    u = (t in usd ? usd[t] : 0)
    f = cfg[t] - u
    if (f<0) f=0
    if (!ok) f=0
    tot_cfg[t]  += cfg[t]
    tot_used[t] += u
    tot_free[t] += f
    if (f>0) nfree[t]++
    if (verbose && (t in wantset) && f>0)
      printf "  %-6s %-16s free=%-2d (cfg=%d used=%d) %s\n", t, node, f, cfg[t], u, state
  }
}
END{
  if (emit){                                       # machine mode: "<type> <free>" only
    if (n>0){ for (i=1;i<=n;i++) printf "%s %d\n", want[i], tot_free[want[i]]+0 }
    else    { for (t in tot_cfg)  printf "%s %d\n", t, tot_free[t]+0 }
    exit
  }
  if (verbose) print ""
  printf "%-8s %6s %6s %7s %7s\n","TYPE","FREE","USED","CONFIG","NODES+"
  best=""; bestf=-1
  if (n>0){ for (i=1;i<=n;i++){ pr(want[i]); if (tot_free[want[i]]+0>bestf){bestf=tot_free[want[i]]+0; best=want[i]} } }
  else    { for (t in tot_cfg){ pr(t);       if (tot_free[t]+0>bestf){bestf=tot_free[t]+0; best=t} } }
  if (best!="")
    printf "\nFreest now: %s (%d free)  ->  append:  hydra.launcher.gres=gpu:%s:1\n", best, bestf, best
}
'

if [[ "$emit" == "0" ]]; then
  # Your remaining quota: how many GPUs you are already running, out of the 4gpu_tier cap.
  used_by_me=$(squeue -u "$USER" -h -t RUNNING -o "%b" 2>/dev/null \
               | awk -F: '/gpu/{s+=$NF} END{print s+0}')
  echo
  echo "4gpu_tier cap: 4 GPUs total (gres/gpu=4). You are currently using: ${used_by_me:-0}."
  echo "To auto-pick the freest type and launch, use: tools/run_on_free_gpu.sh"
fi
