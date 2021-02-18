from glob import glob


with open("english_unit.labels", "w") as f:
    f.write("#id\char")
    f.write("\n")



txts = glob('/home/jhjeong/Librispeech_data/test/txt/*.txt')

wow = [' ','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for txt in txts:   
    with open(txt, "r") as f:
        line = f.readline().strip()
        for i in line:
            if i in wow:
                pass
            else:
                wow.append(i)
        
print(wow)

for i, a in enumerate(wow):
    with open("english_unit.labels", "a") as f:
        f.write(str(i))
        f.write("   ")
        f.write(a)
        f.write("\n")