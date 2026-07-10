from collections import defaultdict #a dictionary that returns 0 for missing keys instead of throwing an error

END= "." #our end-of-word marker


def corpus_to_symbols(corpus: dict[str,int])->dict[tuple[str,...],int]:
    symbol_corpus = {}
    for word, freq in corpus.items():
        symbols = tuple(list(word)+[END]) # can also write : symbols = (*word, END)
        symbol_corpus[symbols]=freq
    return symbol_corpus


def count_pairs(symbol_corpus: dict[tuple[str,...]])->dict[tuple[str,str],int]:
    pair_counts = defaultdict(int)
    for symbols, freq in symbol_corpus.items():
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1]) #consecutive pairs
            pair_counts[pair]+= freq
    return pair_counts

def merge_pair(symbol_corpus: dict[tuple[str,...],int], pair: tuple[str,str])->dict[tuple[str,...],int]:
    new_corpus = {}
    merged_symbol = "".join(pair)

    for symbols, freq in symbol_corpus.items():
        new_symbols = []
        i=0
        while (i<len(symbols)):
            if i<len(symbols)-1 and (symbols[i],symbols[i+1])==pair:
                new_symbols.append(merged_symbol)
                i+=2
            else:
                new_symbols.append(symbols[i])
                i+=1
        new_corpus[tuple(new_symbols)] = freq
    return new_corpus


def train_bpe(corpus: dict[str,int], num_merges: int)->list[tuple[str,str]]: #num_merges=how many times to run the loop, Returns a list of merge rules like [('e','s'), ('es','t'), ...]
    symbol_corpus = corpus_to_symbols(corpus)
    merges = [] #to collect our merge rules in order

    for step in range(num_merges):
        pair_counts = count_pairs(symbol_corpus)
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: pair_counts[p]) #find the pair with the highest frequency
        symbol_corpus = merge_pair(symbol_corpus, best_pair) #merge the best pair in the corpus
        merges.append(best_pair)

        print(f"Merge {step+1}: {best_pair} -> '{''.join(best_pair)}'  "
              f"(count={pair_counts[best_pair]})")
        
    return merges


if __name__ == "__main__":
    toy_corpus = {
        "low": 5,
        "lower": 2,
        "newest": 6,
        "widest": 3,
    }

    print("Training BPE on toy corpus...\n")
    learned_merges = train_bpe(toy_corpus,num_merges=4)

    print("\nLearned merge rules:", learned_merges)

    expected = [("e","s"), ("es","t"), ("est", END), ("l","o")]
    print("Matches paper result:", learned_merges == expected)



