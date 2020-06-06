filepath = '../experiments/projects_clean.txt'

list = []

with open(filepath) as fp:
   line = fp.readline()
   while line:
       line.strip()
       x = line.split()
       list.append(float(x[1]))
       line = fp.readline()

print("max: %f min %f" % (max(list), min(list)))

dict = {}

count10 = 0
count20 = 0
count30 = 0
count40 = 0
count50 = 0
count60 = 0
count70 = 0

def percent(value):
    global count10, count20, count30, count40, count50, count60, count70
    if value < 10:
        count10 = count10 + 1
    elif value < 20 and value > 10:
        count20 = count20 + 1
    elif value < 30 and value > 20:
        count30 = count30 + 1
    elif value < 40 and value > 30:
        count40 = count40 + 1
    elif value < 50 and value > 40:
        count50 = count50 + 1
    elif value < 60 and value > 50:
        count60 = count60 + 1
    elif value < 70 and value > 60:
        count70 = count70 + 1

for item in list:
    percent(item)

print("up to 10: %f" % (count10))
print("up to 20: %f" % (count20))
print("up to 30: %f" % (count30))
print("up to 40: %f" % (count40))
print("up to 50: %f" % (count50))
print("up to 60: %f" % (count60))
print("up to 70: %f" % (count70))
