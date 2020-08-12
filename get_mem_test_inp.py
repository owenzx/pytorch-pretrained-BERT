long_file = "long.min_span"


with open(long_file, 'r') as fr:
    longlines = fr.readlines()




count = 0
for i, line in enumerate(longlines):
    if line[0] == '#':
        continue
    if len(line.strip()) > 3:
        count += 1

    if count == 100:
        print("LEN100: LINE%d"%i)

    if count == 1000:
        print("LEN1000: LINE%d"%i)

