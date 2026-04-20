"""Test: for plain_20f_5n with its 18 codewords (16 singletons + 2 doublets),
any sparse component firing on k<=3 features will ~trivially be "one codeword"
most of the time because codewords are almost singletons.

So "13 alive components each fire on 1-3 features, each matches a codeword"
is a weak statement. Let's quantify: under the null "sample random sparse
feature subsets of size 1-3", how often do they form a single codeword?
"""
import numpy as np
from collections import Counter

# Codewords for plain_20f_5n (from our earlier analysis)
cw_20f_5n = {
    0: "cw0", 1: "cw1", 2: "cw2", 3: "cw3", 4: "cw4",
    5: "cw5", 6: "cw6", 7: "cw7", 8: "cw8", 9: "cw9",
    10: "cw10", 11: "cw11", 12: "cw12", 13: "cw13",
    14: "cw1",   # same as feat 1 (both [1,0,1,0,0])
    15: "cw15", 16: "cw5",  # same as feat 5 (both [0,0,1,1,0])
    17: "cw17", 18: "cw18", 19: "cw19",
}
# Non-singletons: cw1={1,14}, cw5={5,16}. So 16 singletons + 2 doubletons.

rng = np.random.RandomState(0)
trials = 10000
# For each k in {1,2,3}, sample k random features; what fraction form one codeword?
for k in [1, 2, 3]:
    counts = 0
    for _ in range(trials):
        feats = rng.choice(20, size=k, replace=False)
        gs = [cw_20f_5n[int(f)] for f in feats]
        if len(set(gs)) == 1:
            counts += 1
    print(f"k={k}: fraction single codeword = {counts/trials:.3f}")

# Same for 20f_2n (3 codeword groups of size 7/7/6)
print("\n\nplain_20f_2n baseline:")
cw_20f_2n = {}
for j in [1,4,7,15,16,17,18]: cw_20f_2n[j] = "A"
for j in [2,3,5,8,9,11,12]:   cw_20f_2n[j] = "B"
for j in [0,6,10,13,14,19]:   cw_20f_2n[j] = "C"
for k in [1, 2, 3, 4, 5]:
    counts = 0
    for _ in range(trials):
        feats = rng.choice(20, size=k, replace=False)
        gs = [cw_20f_2n[int(f)] for f in feats]
        if len(set(gs)) == 1:
            counts += 1
    print(f"k={k}: fraction single codeword = {counts/trials:.3f}")
