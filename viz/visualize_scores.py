from matplotlib import pyplot as plt
import numpy as np
import json 

def make_bars(score_data):

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))

    for lang_idx, lang in enumerate(score_data):
    
        ax = plt.subplot(2,4,lang_idx + 1) # axs[lang_idx]

        cats = list(score_data[lang].keys())
        scores_combined = {
            key: [
                score_data[lang][cat].get(key, 0) for cat in cats
            ] for key in ("rouge", "bleu")
        }

        x = np.arange(len(cats))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for attribute, measurement in scores_combined.items():
            offset = width * multiplier
            rects = ax.bar(
                x + offset, 
                measurement, 
                width, 
                label=attribute, 
                color="red" if attribute == "rouge" else "blue"
            )
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('score')
        ax.set_title(lang)
        ax.set_xticks(x + width, cats)
        ax.set_ylim(0, 0.5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncols=2)
    plt.tight_layout()
    plt.savefig("bar_charts.png")
    plt.show()

def main(score_file="../egs/scores/full_results.json"):

    with open(score_file, 'r') as f:
        score_data = json.load(f)
    
    make_bars(score_data)

if __name__ == "__main__":
    main()

