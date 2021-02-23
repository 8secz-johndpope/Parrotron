import wave
import contextlib

with open("wowwow1.csv", "w") as f:
    f.write("")


with open("/home/jhjeong/jiho_deep/Parrotron/label,csv/test.csv", "r") as f:
    lines = f.readlines()

    for line in lines:
        file_name = line.split(",")[0]
    
        with contextlib.closing(wave.open(file_name,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        
        if duration < 5:
            with open("wowwow1.csv", "a") as ff:
                ff.write(line)

        
