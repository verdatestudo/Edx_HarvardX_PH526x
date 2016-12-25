
# Case Study 1

dna_filename = 'NM207618_2.txt'
protein_filename = 'NM207618_2_translation.txt'

# from table.py
table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def read_seq(filename):
    '''
    Reads and returns the input seq with special characters removed.
    '''
    with open(filename, 'r') as f:
        seq = f.read()

    return ''.join(ch for ch in seq if ch.isalnum())


def translate_dna(seq):
    '''
    Translate a string containing a nucleotide sequence into a string containing the corresponding sequence of amino acid.
    Nucleotides are translated in triplets using the table dictionary; each amino acid 4 is encoded with a string of length 1.
    '''
    protein = ''
    if len(seq) % 3 == 0:
        for idx in range(0, len(seq), 3):
            codon = seq[idx:idx + 3]
            protein += table[codon]
    return protein

def vid():
    '''
    From vid.
    '''
    seq = read_seq(dna_filename)
    prt = read_seq(protein_filename)

    # from NCBI website, use these indices.
    # https://www.ncbi.nlm.nih.gov/nuccore/NM_207618
    # features CDS = 21 to 938 (20 to 938 in Python)
    # remove last codon as it's the "stop" codon.
    translation = translate_dna(seq[20:938])[:-1]
    print(translation == prt)

vid()
