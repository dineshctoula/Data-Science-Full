# syntax 

for i in range(5):
    print(i)



for i in range(1,6):
    print(i)



for i in range(0,10,2):
    print(i) 
    # yesma chae starting 0 hunxa ane differnece chae 2 hunxa




# while loop 

count = 1

while count <= 5:
    print(count)
    count += 1



# break statement 

for i in range(10):
    if i == 5:
        break
    print(i)
    # yesma chae if i==5 hunxa teti velama break hunxa





# continue statement 

for i in range(5):
    if i == 2:
        continue
    print(i)
    # yesma chae 2 aayo bhane teslae skip hunxa ane aru chalxa 





# nested loop 

for i in range(3):
    # suru ma outer loop chalxa
    #  like 0 0, 01, 02, 10, 11, 12
    for j in range(3):
        print(i, j)



for i in range(1,21):
    print(i)



# print even numbers between 1 and 50 

for i in range(2,51,2):
    print(i)
    


# sum of numbers from 1 to 100 

total =0

for i in range(1,101):
    total=total+i

print(total)


# multiplication table 

num=5
for i in range(1,11):
    print(num, "x",i,"=",num*i)



# guessing game 


secret=7
while True:
    # while true ley chae infinite loop chaluxa until yesma break lagxah
    guess=int(input("enter your number:"))

    if guess==secret:
        print("correct")
        break




data=[12,15,20,18,22]
total=0
for i in data:
    total=total+i

average=total/len(data)
print("average is",average)



# find largest numbers in the data 