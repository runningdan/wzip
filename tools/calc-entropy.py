import math
from typing import List, Any

with open("testinputs/rawin.txt", "rb") as inp:
    raw_chars = inp.read()

char_set = list(raw_chars)

def p(v: int) -> float:
    return float(char_set.count(v)) / float(len(char_set))

prob_sum=0
sum = 0
for i in set(char_set):
    prob_sum += p(i)
    sum += p(i)*math.log2(p(i))

print("sum prob " + str(prob_sum))

entropy = math.ceil(-1 * sum)
min_bits = entropy * len(char_set)

print(f"the calculated entropy is ~ {entropy} bits. Entropy before inverting and rounding: {sum}")
print(f"min bits to represent data: ~ {min_bits} bits")
print(f"target compression ratio: ~ {(float(len(char_set)*8)/float(min_bits)) * 100}%")




