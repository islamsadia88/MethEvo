from os import listdir
from os.path import isfile, join

import numpy as np
from tqdm import tqdm
from Config_file import file_data


config = file_data

PSSM_folder = config['PSSM_folder']
SPD3_folder = config['SPD3_folder']

WINDOW_SIZE = 20
STRUCTURAL_WINDOW_SIZE = 3
LOWEST_VAL = 20

data = {}
final_data = {}


def encoded_to_mapping(encoded):
    global data
    data = {}
    with open(encoded) as fp:
        fp.readline()
        for line in fp:
            row = list(map(str.strip, line.split(',')))
            data[row[1].strip('"')] = row[3].strip().strip('"')


def get_PSSM(protein, protein_sz, ind):
    with open(PSSM_folder + protein + '.seq.pssm') as fp:
        # Read 3 lines for the top padding lines
        fp.readline()
        fp.readline()
        fp.readline()

        #print("ind inside PSSM", ind)

        PSSMs = fp.readlines()
        PSSM_list = []
        #print("PSSMs", PSSMs)

        for val in range(ind - WINDOW_SIZE, ind + WINDOW_SIZE + 1):
            #print("now started")
            now = val
            #print("now", now)
            if val < 0 or val >= protein_sz:
                distance = ind - val
                now = ind + distance
            #print("now ended", PSSMs[now])
            row = list(map(float, PSSMs[now].strip().split()[2:2 + 20]))
            PSSM_list.append(row)
        return PSSM_list


def get_profile_bigram(PSSM, flatten = False):
    B = [[0 for x in range(20)] for y in range(20)]
    #print("B inside PSSM bigram", B)
    for p in range(20):
        for q in range(20):
            now = 0
            #print("P Q",p,q)
            #print("PSSM O O", PSSM[0][0])
            #print("PSSM 1 0", PSSM[1][0])
            #print("PSSM len",len(PSSM))
            for k in range(0, WINDOW_SIZE * 2):
               # print("K", PSSM[k+1][q])
                now += PSSM[k][p] * PSSM[k + 1][q]
            B[p][q] = now
        #print("K", k)
        #print("PSSM K P", PSSM[k][p])
        #print("PSSM k", PSSM[k + 1][q])

    #print("returning from profile bigram")

    return np.asarray(B).flatten() if flatten else B


def get_structural_info(protein, protein_sz, ind, flatten=False):
    with open(SPD3_folder + protein + '.seq.spd3') as fp:
        # Read 3 lines for the top padding lines
        fp.readline()

        SSpres = fp.readlines()
        SSpre_list = []

        for val in range(ind - STRUCTURAL_WINDOW_SIZE, ind + STRUCTURAL_WINDOW_SIZE + 1):
            now = val
            if val < 0 or val >= protein_sz:
                distance = ind - val
                now = ind + distance

            row = list(map(float, SSpres[now].strip().split()[3:]))
            SSpre_list.append(row)
        #print("returning from spd3")
        return np.asarray(SSpre_list).flatten() if flatten else SSpre_list


def get_bigrams(protein, seq, site_str):
    ind = 0

    X = []
    Y = []
    while ind < len(seq):
        ind = seq.find(site_str, ind)
        #print("ind", ind)
        if ind == -1:
            break

        PSSM = get_PSSM(protein, len(seq), ind)
        #print("PSSM done")
        PSSM_bigram = get_profile_bigram(PSSM, True)  # 20 x 20
        #print("PSSM bigram done")
        SPpre = get_structural_info(protein, len(seq), ind, True)
        ind += 1

        combined = np.concatenate((PSSM_bigram, SPpre))

        X.append(combined)
        #X.append(PSSM)
        Y.append(int(site_str))
    #print("returning from bigram")
    return [X, Y]


def main(encoded_file, data_folder, ext, output):
    global data

    print("This is main")

    encoded_to_mapping(encoded_file)
    print("encoded")
    #proteins = [f[0:-9] for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith(ext)]
    proteins = []
    for f in listdir(data_folder):
        #print("lisdir value", f)
        if isfile(join(data_folder, f)) and f.endswith(ext):
            proteins.append(f[0:-9])
            #print("appended", proteins)

    print(proteins)

    X_p = []
    Y_p = []
    X_n = []
    Y_n = []
    count = 0
    for protein in tqdm(proteins):
        count = count + 1
        #print("count", count)
        #print("tqdm values", protein)
        #print("methmetical started")
        #print("data protein value", data)
        mathematical_seq = data[protein].strip()
        #print("math seq", mathematical_seq)

        #print("matmetical done. positive start")
        # For all Positive Sites
        [a, b] = get_bigrams(protein, mathematical_seq, '1')
        X_p += a
        Y_p += b
        #print("for positive sites")

        # For all Negative Sites
        [c, d] = get_bigrams(protein, mathematical_seq, '0')
        X_n += c
        Y_n += d
        #print("for negative sites")
    print("Before saving")
    print("The length")
    print("The length")
    print(len(Y_p))
    print(len(Y_n))

    np.savez('../Results_regeneration/Whole dataset/Our model (PSSM+SPD)/whole_data_features.npz'.format(output), X_p, Y_p, X_n, Y_n)

    print("saved")

    #data1 = np('features_test.npz')
    #lst = data1.files
    #for item in lst:
    #    print(item)
    #    print(data[item])
    #print("printed")


main(config['encoded_file'], config['data_folder'], 'hsb2', config['output'])
