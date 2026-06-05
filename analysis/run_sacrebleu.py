import sacrebleu
import argparse
from collections import Counter

def read_file(filename, tokenizer, lc):
    token_freq = Counter()
    sent = []
    with open(filename, "r") as F:
        for line in F:
            line = line.strip().lower() if lc else line.strip()
            sent.append(line)
            token_freq.update(tokenizer(line).split())
    return sent, token_freq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True, help="Path to reference file")
    parser.add_argument("--hyp", type=str, required=True, help="Path to hypothesis file")
    parser.add_argument("--tokenize", type=str, default="flores200", help="Tokenization method for BLEU score (default: flores200)")
    parser.add_argument("--do_ter", action="store_true", help="Whether to compute TER score")
    parser.add_argument("--lc", default=False, action="store_true", help="Whether to lowercase the input")
    args = parser.parse_args()

    bleu = sacrebleu.metrics.BLEU(smooth_method="none", max_ngram_order=4, tokenize=args.tokenize, lowercase=args.lc)

    refs, token_freq_refs = read_file(args.ref, bleu.tokenizer, args.lc)
    hyps, token_freq_hyps = read_file(args.hyp, bleu.tokenizer, args.lc)

    bleu_score = bleu.corpus_score(hyps, [refs])
    print(f"{bleu_score}\n\t{bleu.get_signature()}\n")

    chrf = sacrebleu.metrics.CHRF(lowercase=args.lc)
    chrf_score = chrf.corpus_score(hyps, [refs])
    print(f"{chrf_score}\n\t{chrf.get_signature()}\n")

    if args.do_ter:
        ter = sacrebleu.metrics.TER(case_sensitive=not(args.lc))
        ter_score = ter.corpus_score(hyps, [refs])
        print(f"{ter_score}\n\t{ter.get_signature()}\n")


    set_refs = set(token_freq_refs.keys())
    set_hyps = set(token_freq_hyps.keys())
    set_both =  set_refs.intersection(set_hyps)
    ref_only = set_refs - set_both
    hyp_only = set_hyps - set_both
    print(f"Ref - most frequent tokens: {token_freq_refs.most_common(20)}")
    print(f"Hyp - most frequent tokens: {token_freq_hyps.most_common(20)}")
    print(f"#types in Ref: {len(set_refs)} / in Hyp: {len(set_hyps)} / in Both: {len(set_both)} ({len(set_both)/len(set_refs):.0%})")
    m1 = min(7, len(ref_only))
    m2 = min(7, len(hyp_only))
    print(f"#types in Ref Only: {len(ref_only)} ({len(ref_only)/len(set_refs):.0%}) e.g. {list(ref_only)[:m1]}")          
    print(f"#types in Hyp Only: {len(hyp_only)} ({len(hyp_only)/len(set_hyps):.0%}) e.g. {list(hyp_only)[:m2]}")