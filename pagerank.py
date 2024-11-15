import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Intialize dictionary mapping pages in corpus to real 0
    prob_dist = {
        state: 0
        for state in corpus
    }
    # Check whether page in corpus
    if page in corpus:
        # Get size of corpus > 0 because page in corpus
        num_pages = len(corpus)
        # Get links on page in corpus because page in corpus
        linked_pages = corpus[page]
        # Find out whether page has links or not
        num_links = len(linked_pages)
        if num_links > 0:
            # Get real probabilities to choose a link on page at random
            for state in linked_pages:
                prob_dist[state] += damping_factor / num_links

            # Get real probabilities to choose any page at random
            for state in corpus:
                prob_dist[state] += (1 - damping_factor) / num_pages
        else:
            # Get real probabilities choosing randomly over all pages equally
            for state in corpus:
                prob_dist[state] += 1 / num_pages
    # Return real probability distribution mapping pages in corpus to real probabilities
    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Get list of pages
    pages = list(corpus.keys())
    # Initialize dictionary mapping each page to number of instances
    samples = {
        page: 0
        for page in pages
    }
    # Randomly choose first sample of a page
    sample = random.choice(pages)
    samples[sample] += 1
    # Randomly choose remaining samples based on transition model of current sample
    for _ in range(1, n):
        model = transition_model(corpus, sample, damping_factor)
        keys = list(model.keys())
        weights = list(model.values())
        sample = random.choices(keys, weights=weights, k=1)[0]
        samples[sample] += 1
    # Make page rank dictionary mapping each page to corresponding proportions
    page_ranks = {
        page: samples[page] / n
        for page in pages
    }
    # Return page rank values
    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize page rank dictionary mapping each page to real probability 0
    page_ranks = {
        page: 0
        for page in corpus
    }
    # Get size of corpus
    num_pages = len(corpus)
    if num_pages > 0:
        # Rank each page equally
        for page in corpus:
            page_ranks[page] = 1 / num_pages
        # Get number of links for each page has into a dictionary
        num_links = {
            page: length if (length := len(corpus[page])) > 0 else num_pages
            for page in corpus
        }
        # Iterately rank every page until accuracy is less than or equal to 0.001
        iterate = True
        while iterate:
            iterate = False
            first_condition = (1 - damping_factor) / num_pages
            for page in corpus:
                current_rank = page_ranks[page]
                second_condition = sum(
                    page_ranks[linking_page] / num_links[linking_page]
                    for linking_page in corpus
                    if page in corpus[linking_page] or len(corpus[linking_page]) == 0
                ) * damping_factor
                new_rank = first_condition + second_condition
                page_ranks[page] = new_rank
                if abs(new_rank - current_rank) > 0.001:
                    iterate = True
    # Return page rank values
    return page_ranks


if __name__ == "__main__":
    main()
