import matplotlib.pyplot as plt
import numpy as np

from utils import *

reference = "CTGCAACGTTCGTGGTTCATGTTTGAGCGATAGGCCGAAACTAACCGTGCATGCAACGTTAGTGGATCATTGTGGAACTATAGACTCAAACTAAGCGAGCTTGCAACGTTAGTGGACCCTTTTTGAGCTATAGACGAAAACGGACCGAGGCTGCAAGGTTAGTGGATCATTTTTCAGTTTTAGACACAAACAAACCGAGCCATCAACGTTAGTCGATCATTTTTGTGCTATTGACCATATCTCAGCGAGCCTGCAACGTGAGTGGATCATTCTTGAGCTCTGGACCAAATCTAACCGTGCCAGCAACGCTAGTGGATAATTTTGTTGCTATAGACCAACACTAATCGAGACTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCCTTACCATCGGACCTCCACGAATCTGAAAAGTTTTAATTTCCGAGCGATACTTACGACCGGACCTCCACGAATCAGAAAGGGTTCACTATCCGCTCGATACATACGATCGGACCTCCACGACTCTGTAAGGTTTCAAAATCCGCACGATAGTTACGACCGTACCTCTACGAATCTATAAGGTTTCAATTTCCGCTGGATCCTTACGATCGGACCTCCTCGAATCTGCAAGGTTTCAATATCCGCTCAATGGTTACGGACGGACCTCCACGCATCTTAAAGGTTAAAATAGGCGCTCGGTACTTACGATCGGACCTCTCCGAATCTCAAAGGTTTCAATATCCGCTTGATACTTACGATCGCAACACCACGGATCTGAAAGGTTTCAATATCCACTCTATA"

query = "CTGCAACGTTCGTGGTTCATGTTTGAGCGATAGGCCGAAACTAACCGTGCATGCAACGTTAGTGGATCATTGTGGAACTATAGACTCAAACTAAGCGAGCTTGCAACGTTAGTGGACCCTTTTTGAGCTATAGACGAAAACGGACCGAGGCTGCAAGGTTAGTGGATCATTTTTCAGTTTTAGACACAAACAAACCGAGCCATCAACGTTAGTCGATCATTTTTGTGCTATTGACCATATCTCAGCGAGCCTGCAACGTGAGTGGATCATTCTTGAGCTCTGGACCAAATCTAACCGTGCCAGCAACGCTAGTGGATAATTTTGTTGCTATAGACCAACACTAATCGAGACTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCCTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCCTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCCTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCCTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCTAGACCAACACTAATCGAGACTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCTAGACCAACACTAATCGAGACTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCTAGACCAACACTAATCGAGACTGCCTCGTTAGTGCATCATTTTTGCGCCATAGACCATAGCTAAGCGAGCGCTCGCTTAGCTATGGTCTATGGCGCAAAAATGATGCACTAACGAGGCAGTCTCGATTAGTGTTGGTCTATAGCAACAAAATTATCCACTAGCGTTGCTGGCTCGCTTAGCTATGGTCTATGGCGCAAAAATGATGCACTAACGAGGCAGTCTCGATTAGTGTTGGTCTATAGCAACAAAATTATCCACTAGCGTTGCTGCTTACCATCGGACCTCCACGAATCTGAAAAGTTTTAATTTCCGAGCGATACTTACGACCGGACCTCCACGAATCAGAAAGGGTTCACTATCCGCTCGATACATACGATCGGACCTCCACGACTCTGTAAGGTTTCAAAATCCGCACGATAGTTACGACCGTACCTCTACGAATCTATAAGGTTTCAATTTCCGCTGGATCCTTACGATCGGACCTCCTCGAATCTGCAAGGTTTCAATATCCGCTCAATGGTTACGGACGGACCTCCACGCATCTTAAAGGTTAAAATAGGCGCTCGGTACTTACGATCGGACCTCTCCGAATCTCAAAGGTTTCAATATCCGCTTGATACTTACGATCGCAACACCACGGATCTGAAAGGTTTCAATATCCACTCTATA"


def main():
    matrix = load_data(reference, query)
    # show_data(matrix)
    pos_chains, neg_chains = get_chains(matrix, 5)
    print(len(reference))
    print(len(query))
    print(len(pos_chains))
    print(len(neg_chains))
    
    chains, start_chains, end_chains, neighbours = get_graph(pos_chains, neg_chains, matrix)
    
    # print("Chains:")
    # for chain in chains:
    #     print(chain.startX, chain.startY, chain.endX, chain.endY, chain.len)
    # print("Neighbours:", neighbours)
    print("Start Finding Path...")
    distance = find_path(chains, start_chains, end_chains, neighbours)
    print("Final Distance:", distance)
    show_chains_effective(pos_chains, neg_chains)


if __name__ == '__main__':
    main()